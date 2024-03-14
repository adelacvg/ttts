import torch
import torch.nn as nn
from einops import rearrange, repeat
from ttts.utils.utils import normalization, AttentionBlock
from ttts.utils.vc_utils import MultiHeadAttention
from ttts.vqvae.modules import LinearNorm, Mish, Conv1dGLU
from ttts.utils import commons

class MelStyleEncoder2(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super(MelStyleEncoder2, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
        )
        self.num_latents = 32
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.hidden_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.cross_attn = MultiHeadAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        max_len = x.shape[1]
        latents = repeat(self.latents, "n d -> b d n", b=x.shape[0])
        x_mask = mask
        latent_lengths = torch.Tensor([self.num_latents for _ in range(x.shape[0])]).to(x.device)
        latent_mask = torch.unsqueeze(commons.sequence_mask(latent_lengths, self.num_latents), 1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * latent_mask.unsqueeze(-1)

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        # self-attention
        if mask is not None:
            x = x * x_mask
        x = self.cross_attn(latents, x, attn_mask)
        x = x.transpose(1, 2)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=torch.zeros(x.shape[0],self.num_latents, dtype=bool).to(x.device))

        return w.unsqueeze(-1)