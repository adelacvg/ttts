import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.vqvae.dataset import PreprocessedMelDataset
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator

from ttts.vqvae.xtts_dvae import DiscreteVAE


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def cycle(dl):
    while True:
        for data in dl:
            yield data
class Trainer(object):
    def __init__(self, cfg_path='ttts/vqvae/config.json'):
        self.accelerator = Accelerator()
        self.cfg = json.load(open(cfg_path))
        self.vqvae = DiscreteVAE(**self.cfg['vqvae'])
        self.dataset = PreprocessedMelDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['dataloader'])
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            # self.ema_model = self._get_target_encoder(self.vqvae).to(self.accelerator.device)
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.ema_updater = EMA(0.999)
        self.optimizer = AdamW(self.vqvae.parameters(),lr=3e-4, betas=(0.9, 0.9999), weight_decay=0.01)
        self.vqvae, self.dataloader, self.optimizer = self.accelerator.prepare(self.vqvae, self.dataloader, self.optimizer)
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.gradient_accumulate_every=1
    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.vqvae),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        vqvae = accelerator.unwrap_model(self.vqvae)
        vqvae.load_state_dict(state_dict)
        # if self.accelerator.is_local_main_process:
        #     self.ema_model.load_state_dict(state_dict)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    mel = next(self.dataloader)
                    mel = mel.to(device).squeeze(1)
                    with self.accelerator.autocast():
                        recon_loss, commitment_loss, mel_recon = self.vqvae(mel)
                        recon_loss = torch.mean(recon_loss)
                        loss = recon_loss+0.25*commitment_loss
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.vqvae)
                accelerator.clip_grad_norm_(self.vqvae.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                #     update_moving_average(self.ema_updater,self.ema_model,self.vqvae)
                if accelerator.is_main_process and self.step % self.val_freq == 0:
                    with torch.no_grad():
                        # self.ema_model.eval()
                        eval_model = self.accelerator.unwrap_model(self.vqvae)
                        eval_model.eval()
                        # mel_recon_ema = self.ema_model.infer(mel)[0]
                        mel_recon_ema = eval_model.infer(mel)[0]
                        eval_model.train()
                    scalar_dict = {"loss": total_loss, "loss_mel":recon_loss, "loss_commitment":commitment_loss, "loss/grad": grad_norm}
                    image_dict = {
                        "all/spec": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred": plot_spectrogram_to_numpy(mel_recon[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred_ema": plot_spectrogram_to_numpy(mel_recon_ema[0, :, :].detach().unsqueeze(-1).cpu()),
                    }
                    summarize(
                        writer=writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq']==0:
                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save(self.step//1000)
                self.step += 1
                pbar.update(1)
        accelerator.print('training complete')


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load('~/tortoise_plus_zh/ttts/vqvae/logs/2023-11-04-00-25-39/model-14.pt')
    trainer.train()
