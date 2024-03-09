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
from ttts.utils.vc_utils import MultiHeadAttention
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
from ttts.vqvae.hifigan import Generator
from ttts.utils import commons


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
                mel_channels=768,
                hubert_channels=100,
                num_heads=8,
                segment_size=32,
                gan_config=None
                ):
        super().__init__()
        self.quantizer = ResidualVectorQuantizer(dimension=dimension, n_q=n_q, bins=bins)
        self.semantic_enc = nn.Sequential(
            nn.Conv1d(mel_channels, hubert_channels, kernel_size=3, padding=1),
            AttentionBlock(hubert_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(hubert_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(hubert_channels, num_heads, relative_pos_embeddings=True),
        )
        self.enc = nn.Sequential(
            nn.Conv1d(hubert_channels, dimension, kernel_size=3, padding=1),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            nn.Conv1d(dimension, dimension, kernel_size=3, stride=2, padding=1),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
        )
        self.ref_proj = nn.Sequential(
            nn.Conv1d(mel_channels, dimension, 1),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
            AttentionBlock(dimension, num_heads, relative_pos_embeddings=True),
        )
        self.ref_enc = RefEncoder(dimension,dimension)
        self.dec = Generator(**gan_config)
        self.segment_size=segment_size
        
    def forward(self, spec, hubert):
        ref = self.ref_proj(spec)
        ref = self.ref_enc(ref)
        x = self.semantic_enc(spec)
        semantic_loss = F.mse_loss(x, hubert.detach(), reduction="mean")
        x = self.enc(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x, layers=[0])
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        quantized = quantized + ref.unsqueeze(-1)
        z_slice, ids_slice = commons.rand_slice_segments(
            quantized, self.segment_size
        )
        o = self.dec(z_slice, g=ref.unsqueeze(-1))
        return o, commit_loss, semantic_loss, ids_slice, quantized
    def infer(self, spec):
        ref = self.ref_proj(spec)
        ref = self.ref_enc(ref)
        x = self.semantic_enc(spec)
        x = self.enc(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x, layers=[0])
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        quantized = quantized + ref.unsqueeze(-1)
        o = self.dec(quantized, g=ref.unsqueeze(-1))
        return o
    def decode(self, code, spec):
        ref = self.ref_proj(spec)
        ref = self.ref_enc(ref)
        quantized = self.quantizer.decode(code.unsqueeze(1))
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        quantized = quantized + ref.unsqueeze(-1)
        out = self.dec(quantized, g=ref)
        return out
    def extract_code(self, spec):
        ref = self.ref_proj(spec)
        ref = self.ref_enc(ref)
        x = self.semantic_enc(spec)
        x = self.enc(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x, layers=[0])
        return codes.transpose(0, 1)