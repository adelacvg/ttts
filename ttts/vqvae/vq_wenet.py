import logging

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
from einops import rearrange, repeat
import copy
import numpy as np
import time
import torch.autograd.profiler as profiler
import math
import torch
from torch import nn
from torch.nn import functional as F
from ttts.utils import commons
from ttts.vqvae import modules, attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm, spectral_norm
from ttts.utils.commons import init_weights, get_padding
from ttts.vqvae.modules import LinearNorm, Mish, Conv1dGLU
from ttts.vqvae.quantize import ResidualVectorQuantizer
from ttts.utils.vc_utils import MultiHeadAttention
from ttts.vqvae.alias_free_torch import *
from ttts.vqvae import activations, monotonic_align


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        gin_channels=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.gin_channels = gin_channels

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        
        self.text_embedding = nn.Embedding(4096, hidden_channels)
        nn.init.normal_(self.text_embedding.weight, 0.0, hidden_channels**-0.5)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, text, text_lengths):
        text_mask = torch.unsqueeze(
            commons.sequence_mask(text_lengths, text.size(1)), 1
        ).to(text.dtype)
        text = self.text_embedding(text)
        text = text.transpose(1, 2)
        text = self.encoder(text * text_mask, text_mask)
        # return text, text_mask

        stats = self.proj(text) * text_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return text, m, logs, text_mask


class SpecEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        sample,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        gin_channels=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.sample = sample
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.gin_channels = gin_channels

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1) 
        if self.gin_channels is not None:
            self.ge_proj = nn.Linear(gin_channels, hidden_channels)
        if self.sample == True:
            self.proj = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, g=None, refer=None, refer_lengths=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        if g is not None:
            y = y + self.ge_proj(g.squeeze(-1)).unsqueeze(
                -1
            )
        if refer is not None:
            y_mask2 = torch.unsqueeze(
                commons.sequence_mask(
                    y_lengths + refer_lengths, y.size(2) + refer.shape[2]
                ),
                1,
            ).to(y.dtype)
            T = y.shape[-1] 
            y = torch.cat([y, refer], dim=2) 
            y = self.encoder(y * y_mask2, y_mask2)
            y = y[:, :, :T] 
        else:
            y = self.encoder(y * y_mask, y_mask)

        y = self.out_proj(y)
        if self.sample == False:
            return y * y_mask

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs


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
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                ) 
            )
            self.flows.append(
                modules.Flip()
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        sample,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.sample = sample

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        if self.sample == True:
            self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask

        x = self.enc(x, x_mask, g=g)

        if self.sample == False:
            return x

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d( 
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(
            ch, 1, 7, 1, padding=3, bias=False
        )
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(
                gin_channels, upsample_initial_channel, 1
            )
            
    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            l.remove_weight_norm()
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.enc = attentions.Encoder(
            filter_channels, filter_channels, 2, 4, kernel_size, p_dropout
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None, dur_detail=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        if dur_detail is not None:
            x = x + dur_detail
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.enc(x * x_mask, x_mask)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        prosody_channels=20,
        n_speakers=0,
        gin_channels=0,
        semantic_frame_rate=None,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

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

        self.t_enc = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            6,
            kernel_size,
            p_dropout,
            latent_channels=192,
            gin_channels=None,
        )

        self.detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            latent_channels=192,
            gin_channels=gin_channels,
        )

        self.refer_feature_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            latent_channels=192,
        )

        self.enc_p = []
        self.enc_p.extend(
            [
                SpecEncoder(
                    inter_channels,
                    hidden_channels,
                    filter_channels,
                    False,
                    n_heads,
                    4,
                    kernel_size,
                    p_dropout,
                    gin_channels=gin_channels,
                ),
                SpecEncoder(
                    inter_channels,
                    hidden_channels,
                    filter_channels,
                    True,
                    n_heads,
                    6,
                    kernel_size,
                    p_dropout,
                    gin_channels=gin_channels,
                ),
            ]
        )
        self.enc_p = nn.ModuleList(self.enc_p)

        self.spec_proj = nn.Conv1d(spec_channels, hidden_channels, 1)

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            True,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        self.ref_enc = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )

        self.dur_detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            latent_channels=192,
            gin_channels=gin_channels,
        )

        self.dur_detail_emb = nn.Embedding(4096, hidden_channels)
        nn.init.normal_(self.dur_detail_emb.weight, 0.0, hidden_channels**-0.5)

        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

    def mas(self, z_p, m_p, logs_p, x_mask, y_mask):
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
        return attn

    def forward(self, y, y_lengths, text, text_lengths):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)
        z, m_q, logs_q = self.enc_q(y, y_lengths, ge)
        z_p = self.flow(z, y_mask, g=ge)

        text, m_t, logs_t, text_mask = self.t_enc(text, text_lengths)

        attn = self.mas(z_p, m_t, logs_t, text_mask, y_mask)
        x_mask = text_mask
        w = attn.sum(2)

        durs = w.squeeze(1).long()
        dur_detail = self.dur_detail_emb(durs).transpose(1, 2)

        dur_detail_ = self.dur_detail_enc(text.detach(), text_lengths, g=ge)

        l_dur_detail = torch.sum(
            ((dur_detail - dur_detail_) ** 2) * x_mask
        ) / torch.sum(x_mask)

        logw = torch.log(w + 1e-6) * x_mask
        logw_ = self.dp(text, x_mask, g=ge, dur_detail=dur_detail)

        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )
        l_length = l_length_dp

        m_t = torch.matmul(attn.squeeze(1), m_t.transpose(1, 2)).transpose(1, 2) # μ
        logs_t = torch.matmul(attn.squeeze(1), logs_t.transpose(1, 2)).transpose(1, 2) # σ
        text = torch.matmul(attn.squeeze(1), text.transpose(1, 2)).transpose(1, 2)

        spec = self.spec_proj(y) * y_mask
        detail = self.enc_p[0](spec, y_lengths, g=ge)
        detail_ = self.detail_enc(text, y_lengths, g=ge)
        l_detail = torch.sum(((detail - detail_) ** 2) * y_mask) / torch.sum(y_mask)

        text = text + detail
        # text = text + detail_ 


        refer_feature = self.refer_feature_enc(spec, y_lengths)

        text, m_p, logs_p = self.enc_p[1](
            text, y_lengths, g=ge, refer=refer_feature, refer_lengths=y_lengths
        )

        quantized = text

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=ge)
        return (
            o,
            ids_slice,
            l_length,
            l_detail,
            l_dur_detail,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q, m_t, logs_t),
            quantized,
        )

    def infer(
        self,
        text,
        text_lengths,
        refer,
        refer_lengths,
        noise_scale=0.667,
        sdp_ratio=0,
        noise_scale_w=0.8,
        length_scale=1.0,
    ):
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1
        ).to(refer.dtype)
        ge = self.ref_enc(refer * refer_mask, refer_mask)

        text, m_t, logs_t, text_mask = self.t_enc(text, text_lengths)
        x_mask = text_mask

        dur_detail = self.dur_detail_enc(text, text_lengths, g=ge)
        logw = self.dp(text, x_mask, g=ge, dur_detail=dur_detail)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask).float()
        text = torch.matmul(attn.squeeze(1), text.transpose(1, 2)).transpose(1, 2)

        detail = self.detail_enc(text, y_lengths, g=ge)
        text = text + detail

        spec = self.spec_proj(refer) * refer_mask
        refer_feature = self.refer_feature_enc(spec, refer_lengths)
        
        x, m_p, logs_p = self.enc_p[1](
            text, y_lengths, g=ge, refer=refer_feature, refer_lengths=refer_lengths
        )

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        o = self.dec(z, g=ge)
        return o
