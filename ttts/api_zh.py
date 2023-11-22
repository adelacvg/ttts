import numpy as np
from pypinyin import lazy_pinyin, Style
import torch
from PIL import Image

from ttts.utils.infer_utils import load_model
from ttts.utils.utils import plot_spectrogram_to_numpy

input_text = "大家好，今天来点大家想看的东西。"
pinyin = ' '.join(lazy_pinyin(input_text, style=Style.TONE3, neutral_tone_with_five=True))

MODELS = {
    'vqvae.pth':'~/tortoise_plus_zh/ttts/vqvae/logs/2023-11-04-15-44-23/model-36.pt',
    'autoregressive.pth': '~/tortoise_plus_zh/ttts/gpt/logs/2023-10-23-16-55-00/model-9.pt',
    'clvp2.pth': '',
    'diffusion_decoder.pth': '~/tortoise_plus_zh/ttts/diffusion/logs/2023-10-27-00-00-28/model-12.pt',
    'vocoder.pth': '~/tortoise_plus_zh/ttts/pretrained_models/pytorch_model.bin',
    'rlg_auto.pth': '',
    'rlg_diffuser.pth': '',
}

import torchaudio
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
device = 'cuda:5'
vqvae = load_model('vqvae',MODELS['vqvae.pth'],'ttts/vqvae/config.json',device)

mel_extractor = MelSpectrogramFeatures().to(device)
audio,sr = torchaudio.load('ttts/0.wav')
audio = torchaudio.transforms.Resample(sr,24000)(audio).to(device)
mel_raw = mel_extractor(audio)

img = plot_spectrogram_to_numpy(mel_raw[0, :, :].detach().unsqueeze(-1).cpu())
image = Image.fromarray(np.uint8(img))
image.save('mel_raw.png')
print(mel_raw.shape)
codes = vqvae.get_codebook_indices(mel_raw)
print(codes)
print(codes.shape)

mel = vqvae.decode(codes)[0]
mel = vqvae.infer(mel_raw)[0]
mel = vqvae(mel)[2]
print(mel.max(),mel.min())
print(mel.shape)
img = plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu())
image = Image.fromarray(np.uint8(img))
image.save('mel_gen.png')
