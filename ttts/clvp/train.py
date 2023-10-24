import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.clvp.dataset import CLVPCollater, CLVPDataset
from ttts.clvp.model import CLVP
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator


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
def warmup(step):
    if step<500:
        return float(step/500)
    else:
        return 1
class Trainer(object):
    def __init__(self, cfg_path='clvp/config.json'):
        self.accelerator = Accelerator()
        self.cfg = json.load(open(cfg_path))
        self.clvp = CLVP(**self.cfg['clvp'])
        self.dataset = CLVPDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['dataloader'], collate_fn=CLVPCollater())
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            self.ema_model = self._get_target_encoder(self.clvp).to(self.accelerator.device)
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.ema_updater = EMA(0.999)
        self.optimizer = AdamW(self.clvp.parameters(),lr=self.cfg['train']['lr'], betas=(0.9, 0.96), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        self.clvp, self.dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(self.clvp, self.dataloader, self.optimizer, self.scheduler)
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.gradient_accumulate_every=self.cfg['train']['accumulate_num']
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
            'model': self.accelerator.get_state_dict(self.ema_model),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(state_dict)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dataloader)
                    if data==None:
                        continue
                    # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
                    input_params = [data['padded_text'], data['padded_qmel']]
                    input_params = [d.to(device) for d in input_params]
                    with self.accelerator.autocast():
                        loss = self.clvp(*input_params, return_loss=True)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.clvp)
                accelerator.clip_grad_norm_(self.clvp.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    update_moving_average(self.ema_updater,self.ema_model,self.clvp)
                if accelerator.is_main_process and self.step % self.val_freq == 0:
                    scalar_dict = {"loss": total_loss, "loss/grad": grad_norm, "lr":self.scheduler.get_last_lr()[0]}
                    summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq']:
                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save(self.step//1000)
                self.step += 1
                pbar.update(1)
        accelerator.print('training complete')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
