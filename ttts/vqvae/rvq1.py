import functools

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, repeat
from ttts.utils.utils import normalization, AttentionBlock
from ttts.utils.vc_utils import MultiHeadAttention
from ttts.diffusion.ldm.util import exists
from ttts.vqvae.quantize import ResidualVectorQuantizer
from ttts.vqvae.hifigan import Generator
from ttts.utils import commons
from ttts.vqvae.modules import WN, Flip, ResidualCouplingLayer

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

class MRTE(nn.Module):
    def __init__(
        self,
        mel_channels=100,
        semantic_channels=1024,
        model_channels=512,
        out_channels=1024,
        gin_channels=512,
        num_heads=4,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(model_channels, model_channels, num_heads)
        self.mel_enc = nn.Sequential(
            nn.Conv1d(mel_channels, model_channels, 3, padding=1),
        )
        self.text_pre = nn.Sequential(
            nn.Conv1d(semantic_channels, model_channels, 1),
        )
        self.c_post = nn.Conv1d(model_channels, semantic_channels, 1)
        self.ge_enc = nn.Sequential(
            nn.Conv1d(gin_channels, model_channels, kernel_size=1),
            )

    def forward(self, refer, text, ge):
        ge = self.ge_enc(ge)
        mel = self.mel_enc(refer)
        text = self.text_pre(text)
        x = (
            self.cross_attention(
                text, mel
            )
            + text
            + ge
        )
        x = self.c_post(x)
        return x

class TextEncoder(nn.Module):
    def __init__(
        self,
        text_channels,
        dim,
        out_channels,
        gin_channels,
        num_layers = 2,
        num_heads = None
    ):
        super().__init__()
        modules=[]
        modules.append(nn.Conv1d(text_channels, dim, kernel_size=3, padding=1))
        for _ in range(num_layers):
            modules.append(AttentionBlock(dim, num_heads, relative_pos_embeddings=True))
        self.enc1 = nn.Sequential(*modules)
        self.mrte = MRTE(
            mel_channels = dim,
            semantic_channels = dim,
            model_channels = dim,
            out_channels =dim,
            gin_channels = gin_channels,
            num_heads=16,
        )
        self.latents = nn.Parameter(torch.randn(256, dim))
        nn.init.normal_(self.latents, std=0.02)
        modules=[]
        for _ in range(num_layers):
            modules.append(AttentionBlock(dim, num_heads, relative_pos_embeddings=True))
        self.enc2 = nn.Sequential(*modules)
        self.proj = nn.Conv1d(dim, out_channels*2, kernel_size=1)
        self.out_channels = out_channels
    def forward(self, x, ge=None):
        x = self.enc1(x)
        latents = repeat(self.latents, "n d -> b d n", b=x.shape[0])
        x = self.mrte(latents, x, ge)
        x = self.enc2(x)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs

class SemanticEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, g=None):
        if g != None:
            g = g.detach()
        x = self.in_proj(x)
        x = self.enc(x, g=g)
        x = self.proj(x)
        return x
class SpecEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, g=None):
        if g != None:
            g = g.detach()
        x = self.in_proj(x)
        x = self.enc(x, g=g)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, g=g, reverse=reverse)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class RVQ1(nn.Module):
    def __init__(self, 
            spec_channels,
            hubert_channels,
            inter_channels,
            dim,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels,
            segment_size=None):
        super().__init__()
        self.semantic_proj = nn.Conv1d(hubert_channels, hubert_channels,3,2,1)
        self.text_enc = TextEncoder(
            hubert_channels, 
            768, inter_channels, gin_channels, 3, 16
        )
        self.semantic_enc = SemanticEncoder(
            spec_channels,
            hubert_channels,
            dim,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.spec_enc = SpecEncoder(
            spec_channels,
            inter_channels,
            dim,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            dim,
            5,
            1,
            4,
            gin_channels=gin_channels
        )
        self.ref_enc = nn.Sequential(
            nn.Conv1d(spec_channels, 1024, 3,2,1),
            RefEncoder(
            1024,
            gin_channels,
            num_latents=16,
            num_heads=16,
        ))
        self.quantizer = ResidualVectorQuantizer(
            dimension=hubert_channels, n_q=1, bins=1024)
        self.segment_size=segment_size
        
    def forward(self, spec, hubert):
        ge = self.ref_enc(spec).unsqueeze(-1)
        semantic = self.semantic_enc(spec,ge)
        semantic_loss = F.l1_loss(hubert.detach(), semantic)
        semantic = self.semantic_proj(semantic)
        quantized, codes, commit_loss, quantized_list = self.quantizer(
            semantic, layers=[0]
        )
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )
        x, m_p, logs_p = self.text_enc(
            quantized, ge
        )
        z, m_q, logs_q = self.spec_enc(spec, g=ge)
        z_p = self.flow(z, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, self.segment_size
        )
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
            semantic_loss
        )

    def infer(self, spec, hubert, noise_scale=0.5):
        ge = self.ref_enc(spec).unsqueeze(-1)

        semantic = self.semantic_enc(spec, ge)
        semantic = self.semantic_proj(semantic)
        quantized, codes, commit_loss, _ = self.quantizer(semantic, layers=[0])
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )

        x, m_p, logs_p = self.text_enc(
            quantized, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, g=ge, reverse=True)

        o = self.dec(z, g=ge)
        return o, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def decode(self, codes, refer_spec, noise_scale=0.5):
        ge = self.ref_enc(refer_spec)
        quantized = self.quantizer.decode(codes)
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )
        x, m_p, logs_p, y_mask = self.text_enc(
            quantized, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, g=ge, reverse=True)

        o = self.dec(z, g=ge)
        return o
    def extract_code(self, spec):
        ge = self.ref_enc(spec).unsqueeze(-1)
        semantic = self.semantic_enc(spec,ge)
        semantic = self.semantic_proj(semantic)
        quantized, codes, commit_loss, _ = self.quantizer(semantic, layers=[0])
        return codes.transpose(0, 1)
