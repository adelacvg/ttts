from ttts.diffusion.mrte import MRTE
import torch as th
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torch import autocast
import math

from ttts.utils.utils import normalization, AttentionBlock

def is_latent(t):
    return t.dtype == torch.float


def is_sequence(t):
    return t.dtype == torch.long


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = ResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb, refer=None):
        y = self.resblk(x, time_emb)
        if refer!=None:
            y = torch.cat([y,refer],dim=-1)
        y = self.attn(y)
        if refer!=None:
            y = y[:,:,:-refer.shape[-1]]
        return y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaseModel(nn.Module):
    def __init__(
        self,
        model_channels,
        out_channels,
        num_layers,
        num_heads=8,
        dropout=0.1,
        referencenet=False
    ):
        super().__init__()
        if referencenet==False:
            self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [ResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])
        else:
            self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)])
        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 3, padding=1),
        )
    def forward(self, x, time_emb=None, latent=None, refers=None,  **kwargs):
        for i, lyr in enumerate(self.layers):
            if i<len(self.layers)-3:
                refer = refers.pop(0)
                x = lyr(x, time_emb, refer=refer)
            else:
                x = lyr(x, time_emb)

        out = self.out(x)
        return out

class ReferenceNet(BaseModel):
    def forward(self, x, time_emb=None, **kwargs):
        refers = []
        for i, lyr in enumerate(self.layers):
            refers.append(x)
            x = lyr(x, time_emb)

        return refers

TACOTRON_MEL_MAX = 5.5451774444795624753378569716654
TACOTRON_MEL_MIN = -16.118095650958319788125940182791
# TACOTRON_MEL_MIN = -11.512925464970228420089957273422
def denormalize_tacotron_mel(norm_mel):
    return norm_mel/0.18215
def normalize_tacotron_mel(mel):
    mel = torch.clamp(mel, min=-TACOTRON_MEL_MAX)
    return mel*0.18215

class AA_diffusion(nn.Module):
    def __init__(self,
        in_channels= 100,
        out_channels= 200,
        model_channels= 512,
        num_heads= 8,
        num_layers= 6,
        in_latent_channels= 512,
        dropout= 0.1,
        mrte=None
        ):
        super().__init__()
        self.model_channels = model_channels
        self.refer_model = ReferenceNet(model_channels, out_channels,
                num_layers, num_heads, dropout, referencenet=True)
        self.base_model = BaseModel(model_channels, out_channels, 
                num_layers, num_heads, dropout)
        print("base model params:", count_parameters(self.base_model))
        self.unconditioned_percentage = 0.1
        self.unconditioned_cat_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.latent_enc = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.refer_enc = nn.Sequential(
            nn.Conv1d(in_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.mrte = MRTE(**mrte)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
        )
        self.inp_block = nn.Conv1d(in_channels, model_channels, 3, 1, 1)
        self.integrating_conv = nn.Conv1d(model_channels*2, model_channels, kernel_size=1)
    def get_uncond_batch(self, code_emb):
        unconditioned_batches = torch.zeros((code_emb.shape[0], 1, 1), device=code_emb.device)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_cat_embedding.repeat(code_emb.shape[0], 1, 1),
                                   code_emb)
        return code_emb
    def forward(self, x, timesteps, latent, refer, conditioning_free=False):
        latent = self.latent_enc(latent)
        refer = self.refer_enc(refer)
        latent = self.mrte(refer, latent)
        if conditioning_free:
            latent = self.unconditioned_cat_embedding.repeat(x.shape[0], 1, x.shape[-1])
        else:
            if self.training:
                latent = self.get_uncond_batch(latent)
            latent = F.interpolate(latent, size=x.shape[-1], mode='nearest')
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        latent = self.conditioning_timestep_integrator(latent, time_emb)
        x = self.inp_block(x)
        x = torch.cat([x, latent], dim=1)
        x = self.integrating_conv(x)
        refers = self.refer_model(refer, time_emb = time_emb)
        eps = self.base_model(x, time_emb=time_emb, latent=latent, refers=refers)
        return eps
