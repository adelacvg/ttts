import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import clean_checkpoints, plot_spectrogram_to_numpy, summarize
from ttts.vqvae.dataset import VQGANDataset, VQVAECollater, BucketSampler
import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
from ttts.vqvae.rvq1 import RVQ1
from ttts.vqvae.gsv_vq import SynthesizerTrn
from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams
from ttts.utils import commons
import torchaudio
from ttts.vqvae.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from ttts.vqvae.hifigan import MultiPeriodDiscriminator

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            else:
                continue
        except:
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def cycle(dl):
    while True:
        for data in dl:
            yield data
class Trainer(object):
    def __init__(self, cfg_path='ttts/vqvae/config.json'):
        self.accelerator = Accelerator()
        self.cfg = json.load(open(cfg_path))
        hps = HParams(**self.cfg)
        self.hps = hps
        dataset = VQGANDataset(hps)
        eval_dataset = VQGANDataset(hps)
        train_sampler = BucketSampler(
            dataset, hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000,
                1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,],
            shuffle=True,)
        collate_fn=VQVAECollater()
        self.dataloader = DataLoader(
            dataset,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=16,)
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.G = SynthesizerTrn(hps.data.filter_length // 2 + 1,hps.train.segment_size // hps.data.hop_length, **hps.vqvae)
        self.D = MultiPeriodDiscriminator()
        print("G params:", count_parameters(self.G))
        print("D params:", count_parameters(self.D))
        self.G_optimizer = AdamW(self.G.parameters(),lr=3e-4, betas=(0.9, 0.9999), weight_decay=0.01)
        self.D_optimizer = AdamW(self.D.parameters(),lr=3e-4, betas=(0.9, 0.9999), weight_decay=0.01)
        self.G, self.G_optimizer, self.D, self.D_optimizer, self.dataloader = self.accelerator.prepare(
            self.G, self.G_optimizer, self.D, self.D_optimizer, self.dataloader)
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
            'G': self.accelerator.get_state_dict(self.G),
            'D': self.accelerator.get_state_dict(self.D),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        G_state_dict = data['G']
        D_state_dict = data['D']
        self.step = data['step']
        G = accelerator.unwrap_model(self.G)
        G.load_state_dict(G_state_dict)
        D = accelerator.unwrap_model(self.D)
        D.load_state_dict(D_state_dict)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        hps = self.hps
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
        epoch=-1
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                self.dataloader.batch_sampler.epoch=epoch
                epoch = epoch + 1
                for data in self.dataloader:
                # with torch.autograd.detect_anomaly():
                    spec = data['spec'].to(device)
                    hubert = data['hubert'].to(device)
                    wav = data['wav'].to(device)
                    spec_length = data['spec_lengths'].to(device)
                    text = data['text'].to(device)
                    text_length = data['text_lengths'].to(device)
                    with self.accelerator.autocast():
                        (y_hat, kl_ssl, ids_slice, x_mask, z_mask,
                            (z, z_p, m_p, logs_p, m_q, logs_q),
                            stats_ssl,) = self.G(hubert, spec, spec_length, text, text_length)
                        #  ssl, y, y_lengths, text, text_length
                        mel = spec_to_mel_torch(
                            spec,
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                        y_mel = commons.slice_segments(
                            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                        )
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )

                        y = commons.slice_segments(
                            wav, ids_slice * hps.data.hop_length, hps.train.segment_size
                        )  # slice
                        # Discriminator
                        y_d_hat_r, y_d_hat_g, _, _ = self.D(y, y_hat.detach())
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
                    self.accelerator.backward(loss_disc_all)
                    D_grad_norm = get_grad_norm(self.D)
                    accelerator.clip_grad_norm_(self.D.parameters(), 1.0)
                    accelerator.wait_for_everyone()
                    self.D_optimizer.step()
                    self.D_optimizer.zero_grad()
                    accelerator.wait_for_everyone()
                    
                    # Generator
                    with self.accelerator.autocast():
                        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.D(y, y_hat)
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl
                    model = self.accelerator.unwrap_model(self.G)

                    self.accelerator.backward(loss_gen_all)
                    G_grad_norm = get_grad_norm(self.G)
                    accelerator.clip_grad_norm_(self.G.parameters(), 1.0)
                    pbar.set_description(f'G_loss:{loss_gen_all:.4f} D_loss:{loss_disc_all:.4f}')
                    accelerator.wait_for_everyone()
                    self.G_optimizer.step()
                    self.G_optimizer.zero_grad()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and self.step % self.val_freq == 0:
                        with torch.no_grad():
                            eval_model = self.accelerator.unwrap_model(self.G)
                            eval_model.eval()
                            wav_eval, _, _ = eval_model.infer(hubert, spec, spec_length, text, text_length)
                            eval_model.train()
                        scalar_dict = {"gen/loss_gen_all": loss_gen_all, "gen/loss_gen":loss_gen,
                            'gen/loss_fm':loss_fm,'gen/loss_mel':loss_mel,'gen/kl_ssl':kl_ssl,
                            'gen/loss_kl':loss_kl, "norm/G_grad": G_grad_norm, "norm/D_grad": D_grad_norm,
                            'disc/loss_disc_all':loss_disc_all}
                        image_dict = {
                            "mel": plot_spectrogram_to_numpy(y_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            "mel_pred": plot_spectrogram_to_numpy(y_hat_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        }
                        audios_dict = {
                            'gt':wav[0].detach().cpu(),
                            'pred':wav_eval[0].detach().cpu()
                        }
                        milestone = self.step // self.cfg['train']['save_freq'] 
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), 32000)
                        summarize(
                            writer=writer,
                            global_step=self.step,
                            images=image_dict,
                            audios=audios_dict,
                            scalars=scalar_dict,
                            audio_sampling_rate=32000
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
    # trainer.load('/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/2024-03-12-10-57-19/model-1.pt')
    trainer.train()
