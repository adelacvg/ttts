from pypinyin import lazy_pinyin, Style
import torch

from ttts.utils.infer_utils import load_model

input_text = "大家好，今天来点大家想看的东西。"
pinyin = ' '.join(lazy_pinyin(input_text, style=Style.TONE3, neutral_tone_with_five=True))

MODELS = {
    'vqvae.pth':'/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/2023-10-31-02-33-25/model-12.pt',
    'autoregressive.pth': '/home/hyc/tortoise_plus_zh/ttts/gpt/logs/2023-10-23-16-55-00/model-9.pt',
    'clvp2.pth': '',
    'diffusion_decoder.pth': '/home/hyc/tortoise_plus_zh/ttts/diffusion/logs/2023-10-27-00-00-28/model-12.pt',
    'vocoder.pth': '/home/hyc/tortoise_plus_zh/ttts/pretrained_models/pytorch_model.bin',
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
mel = mel_extractor(audio)
print(mel.shape)
codes = vqvae.get_codebook_indices(mel)
print(codes)

mel = vqvae.decode(codes)[0]
print(mel.max(),mel.min())
