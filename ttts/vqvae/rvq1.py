import functools
from math import sqrt

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, repeat
from einops import rearrange
from ttts.utils.utils import normalization, AttentionBlock
from ttts.diffusion.mrte import MultiHeadAttention
from ttts.diffusion.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    normalization,
    zero_module,
    timestep_embedding,
)
from ttts.diffusion.ldm.modules.attention import SpatialTransformer
from ttts.diffusion.ldm.util import exists
from ttts.vqvae.quantize import ResidualVectorQuantizer


def default(val, d):
    return val if val is not None else d

class RefEncoder(nn.Module):
    def __init__(
        self,
        ref_dim,
        dim,
        num_latents=32,
        num_heads=8,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, ref_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.cross_attention = MultiHeadAttention(ref_dim, ref_dim, num_heads)
        self.enc = self.enc = nn.Sequential(
            nn.Conv1d(ref_dim, dim, kernel_size=3, padding=1),
            AttentionBlock(dim, num_heads, relative_pos_embeddings=True),
            AttentionBlock(dim, num_heads, relative_pos_embeddings=True),
        )
    def forward(self, x):
        batch = x.shape[0]
        latents = repeat(self.latents, "n d -> b d n", b=batch)
        latents = self.cross_attention(latents, x)
        latents = torch.cat((latents,x),-1)
        latents = self.enc(latents)
        latents = latents[:,:self.latents.shape[1],:]
        latents = torch.mean(latents,-1)
        return latents
    

class RVQ1(nn.Module):
    def __init__(self,
                dimension=1024, 
                n_q=1, 
                bins=1024,
                in_channels=768,
                out_channels=100,
                num_heads=8,
                ):
        super().__init__()
        self.quantizer = ResidualVectorQuantizer(dimension=dimension, n_q=n_q, bins=bins)
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, dimension, kernel_size=3, stride=2, padding=1),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            nn.Conv1d(dimension, dimension, kernel_size=3, stride=2, padding=1),
        )
        self.ref_proj = nn.Conv1d(out_channels, dimension, 1)
        self.ref_enc = RefEncoder(dimension,dimension)
        self.dec = nn.Sequential(
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            nn.Conv1d(dimension, out_channels, kernel_size=1),
        )
    def forward(self, ssl, mel):
        ref = self.ref_proj(mel)
        ref = self.ref_enc(ref)
        ssl = self.enc(ssl)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 4), mode="nearest")
        quantized = quantized + ref.unsqueeze(-1)
        recon = self.dec(quantized)
        recon_loss = F.mse_loss(mel, recon, reduction="mean")
        return recon_loss, commit_loss, recon
    def decode(self, code):
        quantized = self.quantizer.decode(codes)
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 4), mode="nearest")
        quantized = quantized + ref.unsqueeze(-1)
        out = self.dec(quantized)
        return out
    def extract_code(self, x):
        ssl = self.enc(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl,layers=[0])
        return codes.transpose(0, 1)
