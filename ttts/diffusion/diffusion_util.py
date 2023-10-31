import json
import os
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from ttts.unet1d.embeddings import TextTimeEmbedding
from ttts.unet1d.unet_1d_condition import UNet1DConditionModel
from utils import plot_spectrogram_to_numpy
from vocos import Vocos
from torch import expm1, nn
import torchaudio
import commons as commons
from accelerate import Accelerator
from operations import OPERATIONS_ENCODER
from accelerate import DistributedDataParallelKwargs
import math
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import utils

from tqdm.auto import tqdm
TACOTRON_MEL_MAX = 5.5451774444795624753378569716654
TACOTRON_MEL_MIN = -16.118095650958319788125940182791
# TACOTRON_MEL_MIN = -11.512925464970228420089957273422
# -16.118095650958319788125940182791


def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)

class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dropout=0):
        super().__init__()
        self.layer_norm = LayerNorm(c_in)
        conv = ConvTBC(c_in, c_out, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c_in))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = conv

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        return x

class PhoneEncoder(nn.Module):
    def __init__(self,
      in_channels=128,
      hidden_channels=512,
      out_channels=512,
      n_layers=6,
      p_dropout=0.2,
      last_ln = True):
        super().__init__()
        self.arch = [8 for _ in range(n_layers)]
        self.num_layers = n_layers
        self.hidden_size = hidden_channels
        self.padding_idx = 0
        self.dropout = p_dropout
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        self.pre = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        # self.prompt_proj = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        self.out_proj = ConvLayer(hidden_channels, out_channels, 1, p_dropout)
        if last_ln:
            self.layer_norm = LayerNorm(out_channels)
        self.spk_proj = nn.Conv1d(100,hidden_channels,1)

    def forward(self, src_tokens, lengths, g=None):
        # B x C x T -> T x B x C
        src_tokens = self.spk_proj(src_tokens+g)
        src_tokens = rearrange(src_tokens, 'b c t -> t b c')
        # compute padding mask
        encoder_padding_mask = ~commons.sequence_mask(lengths, src_tokens.size(0)).to(torch.bool)
        # prompt_mask = ~commons.sequence_mask(prompt_lengths, prompt.size(0)).to(torch.bool)
        x = src_tokens

        x = self.pre(x, encoder_padding_mask=encoder_padding_mask)
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        # prompt = self.prompt_proj(prompt, encoder_padding_mask=prompt_mask)
        # encoder layers
        for i in range(self.num_layers):
            x = self.layers[i](x, encoder_padding_mask=encoder_padding_mask)
            # x = x+self.attn_blocks[i](x, prompt, prompt, key_padding_mask=prompt_mask)[0]
        x = self.out_proj(x, encoder_padding_mask=encoder_padding_mask)
        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        x = rearrange(x, 't b c-> b c t')
        return x

class PromptEncoder(nn.Module):
    def __init__(self,
      in_channels=128,
      hidden_channels=256,
      out_channels=512,
      n_layers=6,
      p_dropout=0.2,
      last_ln = True):
        super().__init__()
        self.arch = [8 for _ in range(n_layers)]
        self.num_layers = n_layers
        self.hidden_size = hidden_channels
        self.padding_idx = 0
        self.dropout = p_dropout
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(out_channels)
        self.pre = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        self.out_proj = ConvLayer(hidden_channels, out_channels, 1, p_dropout)

    def forward(self, src_tokens, lengths=None):
        # B x C x T -> T x B x C
        src_tokens = rearrange(src_tokens, 'b c t -> t b c')
        # compute padding mask
        encoder_padding_mask = ~commons.sequence_mask(lengths, src_tokens.size(0)).to(torch.bool)
        x = src_tokens

        x = self.pre(x, encoder_padding_mask=encoder_padding_mask)
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        x = self.out_proj(x, encoder_padding_mask=encoder_padding_mask)

        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        x = rearrange(x, 't b c-> b c t')
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)
class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, kernel_size, dropout):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    if dilation==1:
        padding = kernel_size//2
    else:
        padding = dilation
    self.dilated_conv = ConvLayer(residual_channels, 2 * residual_channels, kernel_size)
    self.conditioner_projection = ConvLayer(n_mels, 2 * residual_channels, 1)
    # self.output_projection = ConvLayer(residual_channels, 2 * residual_channels, 1)
    self.output_projection = ConvLayer(residual_channels, residual_channels, 1)
    self.t_proj = ConvLayer(residual_channels, residual_channels, 1)
    self.drop = nn.Dropout(dropout)

  def forward(self, x, diffusion_step, conditioner,x_mask):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)
    #T B C
    y = x + self.t_proj(diffusion_step.unsqueeze(0))
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    conditioner = self.conditioner_projection(conditioner)
    conditioner = self.drop(conditioner)
    y = self.dilated_conv(y) + conditioner
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    gate, filter_ = torch.chunk(y, 2, dim=-1)
    y = torch.sigmoid(gate) * torch.tanh(filter_)
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    y = self.output_projection(y)
    return y
    # y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    # residual, skip = torch.chunk(y, 2, dim=-1)
    # return (x + residual) / math.sqrt(2.0), skip

class Pre_model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.phoneme_encoder = PhoneEncoder(**self.cfg['phoneme_encoder'])
        print("phoneme params:", count_parameters(self.phoneme_encoder))
        self.prompt_encoder = PromptEncoder(**self.cfg['prompt_encoder'])
        print("prompt params:", count_parameters(self.prompt_encoder))
        dim = self.cfg['phoneme_encoder']['out_channels']
        self.ref_enc = TextTimeEmbedding(100, 100, 1)
    def forward(self,data, g=None):
        mel_recon_padded, mel_padded, mel_lengths, refer_padded, refer_lengths = data
        mel_recon_padded, refer_padded = normalize_tacotron_mel(mel_recon_padded), normalize_tacotron_mel(refer_padded)
        g = self.ref_enc(refer_padded.transpose(1,2)).unsqueeze(-1)
        audio_prompt = self.prompt_encoder(refer_padded,refer_lengths)
        content = self.phoneme_encoder(mel_recon_padded, mel_lengths, g)

        return content, audio_prompt
    def infer(self, data):
        mel_recon_padded, refer_padded, mel_lengths, refer_lengths = data
        mel_recon_padded, refer_padded = normalize_tacotron_mel(mel_recon_padded), normalize_tacotron_mel(refer_padded)
        g = self.ref_enc(refer_padded.transpose(1,2)).unsqueeze(-1)
        audio_prompt = self.prompt_encoder(refer_padded,refer_lengths)
        content = self.phoneme_encoder(mel_recon_padded, mel_lengths, g)
        return content, audio_prompt

class Diffusion_Encoder(nn.Module):
  def __init__(self,
      in_channels=128,
      out_channels=128,
      hidden_channels=256,
      block_out_channels = [128,256,384,512],
      n_heads=8,
      p_dropout=0.2,
      ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.n_heads=n_heads
    self.unet = UNet1DConditionModel(
        in_channels=in_channels+hidden_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        norm_num_groups=8,
        cross_attention_dim=hidden_channels,
        attention_head_dim=n_heads,
        addition_embed_type='text',
        resnet_time_scale_shift='scale_shift',
    )


  def forward(self, x, data, t):
    assert torch.isnan(x).any() == False
    contentvec, prompt, contentvec_lengths, prompt_lengths = data
    prompt = rearrange(prompt,' b c t-> b t c')
    x = torch.cat([x, contentvec], dim=1)

    prompt_mask = commons.sequence_mask(prompt_lengths, prompt.size(1)).to(torch.bool)
    x = self.unet(x, t, prompt, encoder_attention_mask=prompt_mask)

    return x.sample

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class Diffuser(nn.Module):
    def __init__(self,
            cfg,
            ddim_sampling_eta = 0,
            min_snr_loss_weight = False,
            min_snr_gamma = 5,
            conditioning_free = True,
            conditioning_free_k  = 1.0
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        print("pre params: ", count_parameters(self.pre_model))
        self.diff_model = Diffusion_Encoder(**cfg['diffusion'])
        print("diff params: ", count_parameters(self.diff_model))
        self.dim = self.diff_model.in_channels
        timesteps = cfg['train']['timesteps']

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = timesteps

        self.unconditioned_content = nn.Parameter(torch.randn(1,cfg['phoneme_encoder']['out_channels'],1))

        # self.sampling_timesteps = cfg['train']['sampling_timesteps']
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)
        self.conditioning_free = conditioning_free
        self.conditioning_free_k  = conditioning_free_k
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data = None):
        model_output = self.diff_model(x,data, t)
        t = t.type(torch.int64) 
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    def sample_fun(self, x, t, data = None):
        if self.conditioning_free:
            # data[1] = self.unconditioned_refer[]
            model_output_no_conditioning = self.diff_model(x, data, t)
        model_output = self.diff_model(x,data, t)
        t = t.type(torch.int64) 
        pred_noise = model_output
        if self.conditioning_free:
            cfk = self.conditioning_free_k
            model_output = (1 + cfk) * model_output - cfk * model_output_no_conditioning

        return pred_noise

    def p_mean_variance(self, x, t, data):
        preds = self.model_predictions(x, t, data)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, data=data)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, content, refer, lengths, refer_lengths, f0, uv, auto_predict_f0 = True):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer = self.pre_model.infer(data)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device = shape[0], refer.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, (content,refer,lengths,refer_lengths))
            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def ddim_sample(self, content, refer, lengths, refer_lengths, f0, uv, auto_predict_f0 = True):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, (content,refer,lengths,refer_lengths))

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def sample(self,
        mel_recon, refer, lengths, refer_lengths,
        # c, refer, f0, uv, lengths, refer_lengths, vocos,
         sampling_timesteps=100, sample_method='unipc'
        ):
        mel_recon, refer = normalize_tacotron_mel(mel_recon), normalize_tacotron_mel(refer)
        if refer.shape[0]==2:
            refer = refer[0].unsqueeze(0)
        self.sampling_timesteps = sampling_timesteps
        if sample_method == 'ddpm':
            sample_fn = self.p_sample_loop
            # audio = sample_fn(c, refer, lengths, refer_lengths, f0, uv, auto_predict_f0)
        elif sample_method == 'ddim':
            sample_fn = self.ddim_sample
            # audio = sample_fn(c, refer, lengths, refer_lengths, f0, uv, auto_predict_f0)
        elif sample_method == 'dpmsolver':
            from sampler.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            def my_wrapper(fn):
                def wrapped(x, t, **kwargs):
                    ret = fn(x, t, **kwargs)
                    self.bar.update(1)
                    return ret

                return wrapped

            # data = (c, refer, f0, 0, 0, lengths, refer_lengths, uv)
            # content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
            shape = (content.shape[1], self.dim, content.shape[0])
            batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            audio = torch.randn(shape, device = device)
            model_fn = model_wrapper(
                my_wrapper(self.sample_fun),
                noise_schedule,
                model_type="x_start",  #"noise" or "x_start" or "v" or "score"
                model_kwargs={"data":(content,refer,lengths,refer_lengths)}
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

            steps = 40
            self.bar = tqdm(desc="sample time step", total=steps)
            audio = dpm_solver.sample(
                audio,
                steps=steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            self.bar.close()
        elif sample_method =='unipc':
            from ttts.sampler.uni_pc import NoiseScheduleVP, model_wrapper, UniPC
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

            def my_wrapper(fn):
                def wrapped(x, t, **kwargs):
                    ret = fn(x, t, **kwargs)
                    self.bar.update(1)
                    return ret

                return wrapped

            data = (mel_recon, refer, lengths, refer_lengths)
            content, refer = self.pre_model.infer(data)
            shape = (content.shape[0], self.dim, content.shape[2])
            batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            audio = torch.randn(shape, device = device)
            model_fn = model_wrapper(
                my_wrapper(self.sample_fun),
                noise_schedule,
                model_type="noise",  #"noise" or "x_start" or "v" or "score"
                model_kwargs={"data":(content,refer,lengths,refer_lengths)}
            )
            uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')
            steps = 30
            self.bar = tqdm(desc="sample time step", total=steps)
            mel = uni_pc.sample(
                audio,
                steps=steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            self.bar.close()

        # mel = audio
        # vocos.to(audio.device)
        # audio = vocos.decode(audio)

        # if audio.ndim == 3:
        #     audio = rearrange(audio, 'b 1 n -> b n')

        # return denormalize(mel)
        return denormalize_tacotron_mel(mel)

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, data, conditioning_free=False):
        unused_params = []
        mel_recon_padded, mel_padded, mel_lengths, refer_padded, refer_lengths = data
        mel_recon_padded, mel_padded = normalize_tacotron_mel(mel_recon_padded), normalize_tacotron_mel(mel_recon_padded)
        assert mel_recon_padded.shape[2] == mel_padded.shape[2]
        b, d, n, device = *mel_padded.shape, mel_padded.device
        x_mask = torch.unsqueeze(commons.sequence_mask(mel_lengths, mel_padded.size(2)), 1).to(mel_padded.dtype)
        x_start = mel_padded*x_mask
        # get pre model outputs
        content, refer = self.pre_model(data)

        if conditioning_free==True:
            refer = self.unconditioned_refer.repeat(data[0].shape[0], 1 ,1) + refer.mean()*0
        else:
            unused_params.append(self.unconditioned_refer)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        noise = torch.randn_like(x_start)*x_mask
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # predict and take gradient step
        model_out = self.diff_model(x,(content,refer,mel_lengths,refer_lengths), t)
        target = noise

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss_diff = reduce(loss, 'b ... -> b (...)', 'mean')
        loss_diff = loss_diff * extract(self.loss_weight, t, loss.shape)
        loss_diff = loss_diff.mean()

        loss = loss_diff

        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        loss = loss + extraneous_addition * 0

        return loss

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
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)