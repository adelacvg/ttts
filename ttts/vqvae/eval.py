from ttts.utils import vc_utils as utils
import torchaudio
import torchaudio.functional as F
import json
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
from ttts.vqvae.vq2 import SynthesizerTrn
from pypinyin import Style, lazy_pinyin
import torch
device = 'cuda:1'
hps_path='ttts/vqvae/config.json'
hps = HParams(**json.load(open(hps_path)))
net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.vqvae,
    ).to(device)
model, _, _, _ = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s" % hps.train.exp_dir, "G_*.pth"),
            net_g,
        )
wav,sr = torchaudio.load('ttts/6.wav')
tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
text = '没什么，没什么。只是他平时总是站在这里，有点奇怪而已。'
# text = '没错没错，就是这样。'
text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
text = tok.encode(text)
text = torch.LongTensor(text).to(device).unsqueeze(0)
text_length = torch.LongTensor([text.shape[-1]]).to(device)

wav32k = F.resample(wav, sr, 32000).to(device)
wav32k = wav32k[:,:int(hps.data.hop_length * 2 * (wav32k.shape[-1]//hps.data.hop_length//2))]
wav32k = torch.clamp(wav32k, min=-1.0, max=1.0)
wav_length = torch.LongTensor([wav32k.shape[-1]]).to(device)
spec = spectrogram_torch(wav32k, hps.data.filter_length,
                hps.data.hop_length, hps.data.win_length, center=False).squeeze(0)
spec_length = torch.LongTensor([
                x//hps.data.hop_length for x in wav_length]).to(device)
with torch.no_grad():
    wav_recon = model.infer(wav32k, wav_length, spec.unsqueeze(0), spec_length, text, text_length, noise_scale=0.5)
torchaudio.save('gen.wav', wav_recon[0].detach().cpu(), 32000)