import torchaudio
from ttts.diffusion.diffusion_util import denormalize_tacotron_mel, normalize_tacotron_mel
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
import torch
import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.diffusion.dataset import DiffusionDataset, DiffusionCollater
from ttts.diffusion.model import DiffusionTts
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
import functools
import random

import torch
from torch.cuda.amp import autocast

from ttts.utils.diffusion import get_named_beta_schedule
from ttts.utils.resample import create_named_schedule_sampler, LossAwareSampler, DeterministicSampler, LossSecondMomentResampler
from ttts.utils.diffusion import space_timesteps, SpacedDiffusion
# from ttts.diffusion.diffusion_util import Diffuser
# from accelerate import DistributedDataParallelKwargs

def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[1] * 4 # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len, False)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                      model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                     progress=verbose)
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]

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
            pass
            # print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def cycle(dl):
    while True:
        for data in dl:
            yield data
def warmup(step):
    if step<1000:
        return float(step/1000)
    else:
        return 1
class Trainer(object):
    def __init__(self, cfg_path='ttts/diffusion/config.json'):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator()
        self.cfg = json.load(open(cfg_path))
        trained_diffusion_steps = 4000
        self.trained_diffusion_steps = 4000
        desired_diffusion_steps = 200
        self.desired_diffusion_steps = 200
        cond_free_k = 2.
        self.diffuser= SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=False, conditioning_free_k=cond_free_k)
        self.infer_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=True, conditioning_free_k=cond_free_k)
        self.diffusion = DiffusionTts(**self.cfg['diffusion'])
        self.dataset = DiffusionDataset(self.cfg)
        self.dataloader = DataLoader(self.dataset, **self.cfg['dataloader'], collate_fn=DiffusionCollater())
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            self.eval_dataloader = DataLoader(self.dataset, batch_size = 1, shuffle= False, num_workers = 16, pin_memory=True, collate_fn=DiffusionCollater())
            self.eval_dataloader = cycle(self.eval_dataloader)
            self.ema_model = self._get_target_encoder(self.diffusion).to(self.accelerator.device)
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.ema_updater = EMA(0.999)
        self.optimizer = AdamW(self.diffusion.parameters(),lr=self.cfg['train']['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        self.diffusion, self.dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(self.diffusion, self.dataloader, self.optimizer, self.scheduler)
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.gradient_accumulate_every=self.cfg['train']['accumulate_num']
        self.unconditioned_percentage = self.cfg['train']['unconditioned_percentage']
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
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, 'eval'))
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dataloader)
                    if data==None:
                        continue
                    # mel_recon_padded, mel_padded, mel_lengths, refer_padded, refer_lengths
                    x_start = normalize_tacotron_mel(data['padded_mel'].to(device))
                    aligned_conditioning = normalize_tacotron_mel(data['padded_mel_recon'].to(device))
                    conditioning_latent = normalize_tacotron_mel(data['padded_mel_refer'].to(device))
                    t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=device).long().to(device)
                    with self.accelerator.autocast():
                        loss = self.diffuser.training_losses( 
                            model = self.diffusion, 
                            x_start = x_start,
                            t = t,
                            model_kwargs = {
                                "aligned_conditioning": aligned_conditioning,
                                "conditioning_latent": conditioning_latent
                            },
                            )["loss"].mean()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.diffusion)
                accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    update_moving_average(self.ema_updater,self.ema_model,self.diffusion)
                if accelerator.is_main_process and self.step % self.val_freq == 0:
                    scalar_dict = {"loss": total_loss, "loss/grad": grad_norm, "lr":self.scheduler.get_last_lr()[0]}
                    summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq'] == 0:
                    self.ema_model.eval()
                    data = next(self.eval_dataloader)
                    mel_recon_padded, mel_padded, mel_lengths,\
                    refer_padded, refer_lengths = data['padded_mel_recon'], data['padded_mel'], data['mel_lengths'], data['padded_mel_refer'], data['mel_refer_lengths']

                    mel_recon_padded, refer_padded = mel_recon_padded.to(device), refer_padded.to(device)
                    lengths, refer_lengths = torch.tensor(mel_recon_padded.size(2),dtype=torch.long).to(device).unsqueeze(0),\
                        torch.tensor(refer_padded.size(2),dtype=torch.long).to(device).unsqueeze(0)
                    mel_recon_padded, refer_padded = normalize_tacotron_mel(mel_recon_padded), normalize_tacotron_mel(refer_padded)
                    with torch.no_grad():
                        # mel = self.ema_model.sample(mel_recon_padded, refer_padded, lengths, refer_lengths)
                        mel = do_spectrogram_diffusion(self.ema_model, self.infer_diffuser,mel_recon_padded,refer_padded,temperature=0.8)
                        mel = mel.detach().cpu()
                    mel_recon_padded = denormalize_tacotron_mel(mel_recon_padded)
                    image_dict = {
                        f"recon/mel": plot_spectrogram_to_numpy(mel_recon_padded[0, :, :].detach().unsqueeze(-1).cpu()),
                        f"gt/mel": plot_spectrogram_to_numpy(mel_padded[0, :, :].detach().unsqueeze(-1).cpu()),
                        f"gen/mel": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                    }
                    summarize(
                        writer=writer_eval,
                        global_step=self.step,
                        images=image_dict,
                    )

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
    trainer.train()
