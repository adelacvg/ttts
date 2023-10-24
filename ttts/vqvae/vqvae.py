# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/rosinality/vq-vae-2-pytorch
# Which was itself borrowed from https://github.com/deepmind/sonnet


import torch
from torch import nn
from torch.nn import functional as F

import torch.distributed as distributed

# from utils.util import checkpoint, opt_get


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, new_return_order=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codes = None
        self.new_return_order = new_return_order

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, return_soft_codes=False):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        soft_codes = -dist
        _, embed_ind = soft_codes.max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        if return_soft_codes:
            return quantize, diff, embed_ind, soft_codes.view(input.shape[:-1] + (-1,))
        elif self.new_return_order:
            return quantize, embed_ind, diff
        else:
            return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, conv_module):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_module(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            conv_module(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, conv_module):
        super().__init__()

        if stride == 4:
            blocks = [
                conv_module(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                conv_module(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                conv_module(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                conv_module(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                conv_module(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, conv_module))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, conv_module, conv_transpose_module
    ):
        super().__init__()

        blocks = [conv_module(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, conv_module))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    conv_transpose_module(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    conv_transpose_module(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                conv_transpose_module(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        codebook_dim=64,
        codebook_size=512,
        conv_module=nn.Conv2d,
        conv_transpose_module=nn.ConvTranspose2d,
        decay=0.99,
    ):
        super().__init__()

        self.unsqueeze_channels = in_channel == -1
        in_channel = abs(in_channel)

        self.codebook_size = codebook_size
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4, conv_module=conv_module)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2, conv_module=conv_module)
        self.quantize_conv_t = conv_module(channel, codebook_dim, 1)
        self.quantize_t = Quantize(codebook_dim, codebook_size)
        self.dec_t = Decoder(
            codebook_dim, codebook_dim, channel, n_res_block, n_res_channel, stride=2, conv_module=conv_module, conv_transpose_module=conv_transpose_module
        )
        self.quantize_conv_b = conv_module(codebook_dim + channel, codebook_dim, 1)
        self.quantize_b = Quantize(codebook_dim, codebook_size)
        self.upsample_t = conv_transpose_module(
            codebook_dim, codebook_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            codebook_dim + codebook_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
            conv_module=conv_module,
            conv_transpose_module=conv_transpose_module
        )

    def forward(self, input):
        if self.unsqueeze_channels:
            input = input.unsqueeze(1)
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        if self.unsqueeze_channels:
            dec = dec.squeeze(1)

        return dec, diff

    def encode(self, input, checkpoint):
        enc_b = checkpoint(self.enc_b, input)
        enc_t = checkpoint(self.enc_t, enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute((0,2,3,1) if len(input.shape) == 4 else (0,2,1))
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute((0,3,1,2) if len(input.shape) == 4 else (0,2,1))
        diff_t = diff_t.unsqueeze(0)

        dec_t = checkpoint(self.dec_t, quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = checkpoint(self.quantize_conv_b, enc_b).permute((0,2,3,1) if len(input.shape) == 4 else (0,2,1))
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute((0,3,1,2) if len(input.shape) == 4 else (0,2,1))
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def encode_only_quantized(self, input):
        qt, qb, d, idt, idb = self.encode(input)
        # Append top and bottom into the same sequence, adding the codebook length onto the top to discriminate it.
        idt += self.codebook_size
        ids = torch.cat([idt, idb], dim=1)
        return ids

    def decode(self, quant_t, quant_b, checkpoint):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = checkpoint(self.dec, quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute((0,3,1,2) if len(code_t.shape) == 4 else (0,2,1))
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute((0,3,1,2) if len(code_t.shape) == 4 else (0,2,1))

        dec = self.decode(quant_t, quant_b)

        return dec

    # Performs decode_code() with the outputs from encode_only_quantized.
    def decode_code_joined(self, input):
        b, s = input.shape
        assert s % 3 == 0  # If not, this tensor didn't come from encode_only_quantized.
        s = s // 3

        # This doesn't work with batching. TODO: fixme.
        t = input[:,:s] - self.codebook_size
        b = input[:,s:]
        return self.decode_code(t, b)


# @register_model
# def register_vqvae(opt_net, opt):
#     kw = opt_get(opt_net, ['kwargs'], {})
#     vq = VQVAE(**kw)
#     return vq


# @register_model
# def register_vqvae_audio(opt_net, opt):
#     kw = opt_get(opt_net, ['kwargs'], {})
#     kw['conv_module'] = nn.Conv1d
#     kw['conv_transpose_module'] = nn.ConvTranspose1d
#     vq = VQVAE(**kw)
#     return vq


if __name__ == '__main__':
    model = VQVAE(in_channel=80, conv_module=nn.Conv1d, conv_transpose_module=nn.ConvTranspose1d)
    #res=model(torch.randn(1,80,2048))
    e = model.encode_only_quantized(torch.randn(1, 80, 2048))
    k = model.decode_code_joined(e)
    print(k.shape)
