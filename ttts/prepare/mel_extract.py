import os
import torchaudio
import torch
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
device = 'cuda'
mel_extractor = MelSpectrogramFeatures().to(device)

def process_mel(wav_file):
    # print(f"Processing {wav_file}..")
    outfile = f'{wav_file}.mel.pth'
    if os.path.exists(outfile):
        return

    try:
        wave, sample_rate = torchaudio.load(wav_file)
        wave = wave.to(device)
        if wave.size(0) > 1:  # mix to mono
            wave = wave[0].unsqueeze(0)
        if sample_rate!=24000:
            resample = torchaudio.transforms.Resample(sample_rate,24000).to(device)
            wave = resample(wave)
    except Exception as e:
        print(e)
        print(f"Error with {wav_file}")
        return
    mel = mel_extractor(wave)
    torch.save(mel.cpu().detach(), outfile)