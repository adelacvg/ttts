import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.classifier.dataset import PreprocessedMelDataset
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator

from ttts.classifier.model import AudioMiniEncoderWithClassifierHead


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
    def __init__(self, cfg_path='ttts/classifier/config.json'):
        self.accelerator = Accelerator()
        self.cfg = json.load(open(cfg_path))
        self.classifier = AudioMiniEncoderWithClassifierHead(**self.cfg['classifier'])
        self.dataset = PreprocessedMelDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['dataloader'])
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.optimizer = AdamW(self.classifier.parameters(),lr=3e-4, betas=(0.9, 0.9999), weight_decay=0.01)
        self.classifier, self.dataloader, self.optimizer = self.accelerator.prepare(self.classifier, self.dataloader, self.optimizer)
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
            'model': self.accelerator.get_state_dict(self.classifier),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        classifier = accelerator.unwrap_model(self.classifier)
        classifier.load_state_dict(state_dict)
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
                    mel, labels = next(self.dataloader)
                    mel = mel.to(device).squeeze(1)
                    labels = labels.to(device)
                    with self.accelerator.autocast():
                        loss = self.classifier(mel,labels)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.classifier)
                accelerator.clip_grad_norm_(self.classifier.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process and self.step % self.val_freq == 0:
                    scalar_dict = {"loss": total_loss, "loss/grad": grad_norm}
                    summarize(
                        writer=writer,
                        global_step=self.step,
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
    trainer.load('/home/hyc/tortoise_plus_zh/ttts/classifier/logs/2023-11-22-01-14-15/model-4.pt')
    trainer.train()