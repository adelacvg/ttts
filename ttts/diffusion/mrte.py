# This is Multi-reference timbre encoder

import torch
from torch import nn
from ttts.utils.utils import normalization, AttentionBlock
import torch.nn.functional as F
import math
from ttts.vqvae.rvq1 import RefEncoder
from ttts.utils.vc_utils import MultiHeadAttention


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
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.text_pre = nn.Sequential(
            nn.Conv1d(semantic_channels, model_channels, 1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.c_post = nn.Conv1d(model_channels, semantic_channels, 1)
        self.ge_enc = nn.Sequential(
            nn.Conv1d(mel_channels, model_channels, kernel_size=3, padding=1),
            RefEncoder(model_channels,model_channels),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            )
        self.post_enc = nn.Sequential(
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )


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
        x = self.post_enc(x)
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