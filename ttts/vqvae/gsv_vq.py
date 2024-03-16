import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from ttts.utils import commons
from ttts.vqvae import modules, attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from ttts.utils.commons import init_weights, get_padding
from ttts.vqvae.quantize import ResidualVectorQuantizer
from ttts.utils.vc_utils import MultiHeadAttention
from ttts.vqvae.alias_free_torch import *
from ttts.vqvae import activations

class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels=192,
        hidden_size=512,
        out_channels=192,
        kernel_size=5,
        n_heads=4,
        ge_layer=2,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc, ssl_mask, text, text_mask, ge, test=None):
        if ge == None:
            ge = 0
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)

        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = (
            self.cross_attention(
                ssl_enc * ssl_mask, text_enc * text_mask, attn_mask
            )
            + ssl_enc
            + ge
        )
        x = self.c_post(x * ssl_mask)
        return x

class MelDecoder(nn.Module):
  def __init__(self,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      mel_size=20,
      gin_channels=0):
    super().__init__()

    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.conv_pre = Conv1d(hidden_channels, hidden_channels, 3, 1, padding=1)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)

    self.proj= nn.Conv1d(hidden_channels, mel_size, 1, bias=False)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

  def forward(self, x, x_mask, g=None):

    x = self.conv_pre(x*x_mask)
    if g is not None:
        x = x + self.cond(g)

    x = self.encoder(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    return x

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

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        # self.encoder_text = attentions.Encoder(
        #     hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        # )
        # self.text_embedding = nn.Embedding(256, hidden_channels)

        # self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge, test=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )

        y = self.ssl_proj(y * y_mask) * y_mask
     
        y = self.encoder_ssl(y * y_mask, y_mask)

        # text_mask = torch.unsqueeze(
        #     commons.sequence_mask(text_lengths, text.size(1)), 1
        # ).to(y.dtype)
        # if test == 1:
        #     text[:, :] = 0
        # text = self.text_embedding(text).transpose(1, 2)
        # text = self.encoder_text(text * text_mask, text_mask)
        # y = self.mrte(y, y_mask, text, text_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized



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
            self.flows.append(modules.Flip())

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

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class WNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
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

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm = modules.LayerNorm(out_channels)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        out = self.proj(x) * x_mask
        out = self.norm(out)
        return out


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

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

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
            remove_parametrizations(l)
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


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        d = (
            torch.sum(x**2, 1, keepdim=True)
            + torch.sum(self.embedding.weight**2, 1)
            - 2 * torch.matmul(x, self.embedding.weight.T)
        )
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies


class Quantizer(torch.nn.Module):
    def __init__(self, embed_dim=512, n_code_groups=4, n_codes=160):
        super(Quantizer, self).__init__()
        assert embed_dim % n_code_groups == 0
        self.quantizer_modules = nn.ModuleList(
            [
                Quantizer_module(n_codes, embed_dim // n_code_groups)
                for _ in range(n_code_groups)
            ]
        )
        self.n_code_groups = n_code_groups
        self.embed_dim = embed_dim

    def forward(self, xin):
        # B, C, T
        B, C, T = xin.shape
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, self.embed_dim)
        x = torch.split(x, self.embed_dim // self.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        for _x, m in zip(x, self.quantizer_modules):
            _z_q, _min_indicies = m(_x)
            z_q.append(_z_q)
            min_indicies.append(_min_indicies)  # B * T,
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean(
            (z_q - xin.detach()) ** 2
        )
        z_q = xin + (z_q - xin).detach()
        z_q = z_q.transpose(1, 2)
        codes = torch.stack(min_indicies, -1).reshape(B, T, self.n_code_groups)
        return z_q, loss, codes.transpose(1, 2)

    def embed(self, x):
        # idx: N, 4, T
        x = x.transpose(1, 2)
        x = torch.split(x, 1, 2)
        ret = []
        for q, embed in zip(x, self.quantizer_modules):
            q = embed.embedding(q.squeeze(-1))
            ret.append(q)
        ret = torch.cat(ret, -1)
        return ret.transpose(1, 2)  # N, C, T


class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
      
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers


        self.activations = nn.ModuleList([
            Activation1d(
                activation=activations.SnakeBeta(channels, alpha_logscale=True))
                for _ in range(self.num_layers)
        ])
  
    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class PosteriorAudioEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.down_pre = nn.Conv1d(1, 16, 7, 1, padding=3)
    self.resblocks = nn.ModuleList()
    downsample_rates = [10,8,4,2,2]
    downsample_kernel_sizes = [16, 16, 8, 4, 4]
    ch = [16, 32, 64, 96, 128, 192]

    resblock = modules.ResBlock1
    # resblock = AMPBlock1
    resblock_kernel_sizes = [3,7,11]
    resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
    self.num_kernels = 3
    self.downs = nn.ModuleList()
    for i, (u, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
        self.downs.append(weight_norm(
            Conv1d(ch[i], ch[i+1], k, u, padding=(k-1)//2)))
    for i in range(5):
        for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.resblocks.append(resblock(ch[i+1], k, d))

    activation_post = activations.SnakeBeta(ch[i+1], alpha_logscale=True)
    self.activation_post = Activation1d(activation=activation_post)

    self.conv_post = Conv1d(ch[i+1], hidden_channels, 7, 1, padding=3)


    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels*2, out_channels * 2, 1)

  def forward(self, x, x_audio, x_mask, g=None):

    x_audio = self.down_pre(x_audio)

    for i in range(5):

        x_audio = self.downs[i](x_audio)

        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i*self.num_kernels+j](x_audio)
            else:
                xs += self.resblocks[i*self.num_kernels+j](x_audio)
        x_audio = xs / self.num_kernels

    x_audio = self.activation_post(x_audio)
    x_audio = self.conv_post(x_audio)

    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)

    assert x_audio.shape[-1]==x_mask.shape[-1]
    x_audio = x_audio * x_mask

    x = torch.cat([x, x_audio], dim=1)

    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

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
        prosody_size=20,
        n_speakers=0,
        gin_channels=0,
        semantic_frame_rate=None,
        freeze_quantizer=None,
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
        self.mel_size = prosody_size

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
        print("dec:", count_parameters(self.dec))
        # self.mel_decoder = MelDecoder(
        #     inter_channels, filter_channels, n_heads=2,
        #     n_layers=2, kernel_size=5, p_dropout=0.1,
        #     mel_size=self.mel_size, gin_channels=gin_channels)
        # print("mel_dec:", count_parameters(self.mel_decoder))
        self.enc_p = PosteriorAudioEncoder(
            spec_channels, inter_channels, hidden_channels,
            5, 1, 16, gin_channels=gin_channels)
        # print("enc_p:", count_parameters(self.enc_p))
        self.enc_p_2 = PosteriorEncoder(
            inter_channels, inter_channels, hidden_channels,
            5, 1, 16, gin_channels=gin_channels)
        # print("enc_p_2:", count_parameters(self.enc_p_2))
        self.enc_q = PosteriorAudioEncoder(
            spec_channels, inter_channels, hidden_channels,
            5, 1, 16, gin_channels=gin_channels)
        # print("enc_q:", count_parameters(self.enc_q))
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.ref_enc = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )
        self.quantizer = ResidualVectorQuantizer(dimension=inter_channels, n_q=1, bins=1024)
        # if freeze_quantizer:
        #     self.ssl_proj.requires_grad_(False)
        #     self.quantizer.requires_grad_(False)

    def forward(self, wav, wav_aug, wav_lengths, y, y_aug, y_lengths, text, text_lengths):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)

        x, _, _ = self.enc_p(y_aug, wav_aug.unsqueeze(1), y_mask, g=ge)
        quantized, codes,\
        commit_loss, quantized_list = self.quantizer(x, layers=[0])
        x, m_p, logs_p = self.enc_p_2(quantized, y_lengths, g=ge)
        z, m_q, logs_q = self.enc_q(y, wav.unsqueeze(1), y_mask, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, wav, wav_lengths, y, y_lengths, text, text_lengths, noise_scale=0.5):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)

        x, _, _ = self.enc_p(y, wav.unsqueeze(1), y_mask, g=ge)
        quantized, codes,\
        commit_loss, quantized_list = self.quantizer(x, layers=[0])
        x, m_p, logs_p = self.enc_p_2(quantized, y_lengths, g=ge)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def decode(self, codes, text, refer, noise_scale=0.5):
        ge = None
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1
        ).to(refer.dtype)
        ge = self.ref_enc(refer * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([codes.size(2)]).to(codes.device)
        quantized = self.quantizer.decode(codes)
        x, m_p, logs_p, y_mask = self.enc_p_2(
            quantized, y_lengths, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o

    def extract_latent(self, wav, y):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)
        x, _, _ = self.enc_p(y, wav.unsqueeze(1), y_mask, g=ge)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)