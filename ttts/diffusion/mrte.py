# This is Multi-reference timbre encoder
import torch
from torch import nn
from ttts.utils.utils import normalization, AttentionBlock
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from ttts.utils.vc_utils import MultiHeadAttention

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

class MRTE(nn.Module):
    def __init__(
        self,
        mel_channels=100,
        semantic_channels=1024,
        model_channels=512,
        out_channels=1024,
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
            nn.Conv1d(mel_channels, model_channels, kernel_size=3, padding=1),
            RefEncoder(model_channels,model_channels),
            # AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            # AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            )
        # self.post_enc = nn.Sequential(
        #     AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        #     AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        # )


    def forward(self, refer, text):
        ge = self.ge_enc(refer)
        mel = self.mel_enc(refer)
        text = self.text_pre(text)
        x = (
            self.cross_attention(
                text, mel
            )
            + text
            + ge.unsqueeze(-1)
        )
        x = self.c_post(x)
        return x

if __name__ == "__main__":
    content_enc = torch.randn(3, 192, 100)
    content_mask = torch.ones(3, 1, 100)
    ref_mel = torch.randn(3, 128, 30)
    ref_mask = torch.ones(3, 1, 30)
    model = MRTE()
    out = model(content_enc, content_mask, ref_mel, ref_mask)
    print(out.shape)