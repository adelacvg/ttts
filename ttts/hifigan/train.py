import copy
from datetime import datetime
from inspect import signature
import json
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm
from ttts.diffusion.diffusion_util import cycle, get_grad_norm, normalize_tacotron_mel
from ttts.diffusion.train import set_requires_grad
from ttts.hifigan.dataset import HiFiGANCollater, HifiGANDataset
from torch.utils.tensorboard import SummaryWriter
from ttts.hifigan.hifigan_discriminator import HifiganDiscriminator
from ttts.hifigan.hifigan_vocoder import HifiDecoder
from ttts.hifigan.losses import DiscriminatorLoss, GeneratorLoss
from ttts.utils.infer_utils import load_model
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize
import torch
from typing import Any, Callable, Dict, Union, Tuple
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import os

from ttts.vocoder.feature_extractors import MelSpectrogramFeatures

def warmup(step):
    if step<1000:
        return float(step/1000)
    else:
        return 1

class Trainer(object):
    def __init__(self, cfg_path='ttts/hifigan/config.json'):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator()
        self.cfg = json.load(open(cfg_path))

        self.gpt = load_model('gpt',self.cfg['dataset']['gpt_path'],'ttts/gpt/config.json','cuda')
        self.mel_length_compression  = self.gpt.mel_length_compression

        self.hifigan_decoder = HifiDecoder(
            **self.cfg['hifigan']
        )
        self.hifigan_discriminator = HifiganDiscriminator()
        self.dataset = HifiGANDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['dataloader'], collate_fn=HiFiGANCollater())
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.G_optimizer = AdamW(self.hifigan_decoder.parameters(),lr=self.cfg['train']['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        self.D_optimizer = AdamW(self.hifigan_discriminator.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        self.G_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_optimizer, lr_lambda=warmup)
        self.D_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_optimizer, lr_lambda=warmup)
        self.hifigan_decoder, self.hifigan_discriminator, self.dataloader, self.G_optimizer, self.D_optimizer, self.G_scheduler, self.D_scheduler, self.gpt = self.accelerator.prepare(self.hifigan_decoder, self.hifigan_discriminator, self.dataloader, self.G_optimizer, self.D_optimizer, self.G_scheduler, self.D_scheduler, self.gpt)
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.mel_extractor = MelSpectrogramFeatures().to(self.accelerator.device)
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.accelerator.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.accelerator.device)
        )
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
            'model': self.accelerator.get_state_dict(self.hifigan_decoder),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        model = self.accelerator.unwrap_model(self.hifigan_decoder)
        model.load_state_dict(state_dict)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, 'eval'))
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                # 'padded_text': padded_text,
                # 'padded_mel_code': padded_mel_code,
                # 'padded_wav': padded_wav,
                # 'padded_wav_refer':padded_wav_refer,
                data = next(self.dataloader)
                data = {k: v.to(device) for k, v in data.items()}
                y = data['padded_wav']
                if data==None:
                    continue
                mel_refer = self.mel_extractor(data['padded_wav_refer']).squeeze(1) 
                with torch.no_grad():
                    latent = self.gpt(mel_refer, data['padded_text'],
                        torch.tensor([data['padded_text'].shape[-1]], device=device), data['padded_mel_code'],
                        torch.tensor([data['padded_mel_code'].shape[-1]*self.mel_length_compression], device=device),
                        return_latent=True, clip_inputs=False).transpose(1,2)

                x = latent 
                with self.accelerator.autocast():
                    g = self.get_speaker_embedding(data['padded_wav'], 24000)
                    y_hat = self.hifigan_decoder(x, g)
                    score_fake, feat_fake = self.hifigan_discriminator(y_hat.detach())
                    score_real, feat_real = self.hifigan_discriminator(y.clone())
                    loss_d = self.disc_loss(score_fake, score_real)['loss']

                self.accelerator.backward(loss_d)
                grad_norm_d = get_grad_norm(self.hifigan_discriminator)
                accelerator.clip_grad_norm_(self.hifigan_discriminator.parameters(), 1.0)
                accelerator.wait_for_everyone()
                self.D_optimizer.step()
                self.D_optimizer.zero_grad()
                self.D_scheduler.step()
                accelerator.wait_for_everyone()

                score_fake, feat_fake = self.hifigan_discriminator(y_hat)
                loss_g = self.gen_loss(y_hat, y, score_fake, feat_fake, feat_real)['loss']
                self.accelerator.backward(loss_g)
                grad_norm_g = get_grad_norm(self.hifigan_decoder)
                accelerator.clip_grad_norm_(self.hifigan_decoder.parameters(), 1.0)
                accelerator.wait_for_everyone()
                self.G_optimizer.step()
                self.G_optimizer.zero_grad()
                self.G_scheduler.step()
                accelerator.wait_for_everyone()
                    
                pbar.set_description(f'loss_d: {loss_d:.4f} loss_g: {loss_g:.4f}')
                # if accelerator.is_main_process:
                #     update_moving_average(self.ema_updater,self.ema_model,self.diffusion)
                if accelerator.is_main_process and self.step % self.val_freq == 0:
                    scalar_dict = {"loss_d": loss_d, "loss/grad_d": grad_norm_d, "lr_d":self.D_scheduler.get_last_lr()[0],
                                   "loss_g": loss_g, "loss/grad_g": grad_norm_g, "lr_g":self.G_scheduler.get_last_lr()[0],}
                    summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq'] == 0:
                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save(self.step//1000)
                    self.ema_model.train()
                self.step += 1
                pbar.update(1)
        accelerator.print('training complete')


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load('')
    trainer.train()