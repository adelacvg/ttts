import os
import torchaudio
import torch
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from pydub import  AudioSegment
import torchaudio.functional as F
import numpy as np
device = 'cuda'
mel_extractor = MelSpectrogramFeatures().to(device)

def process_mel(wav_file):
    # print(f"Processing {wav_file}..")
    outfile = f'{wav_file}.mel.pth'
    # if os.path.exists(outfile):
    #     return

    try:
        wav, sr = torchaudio.load(wav_file)
        if wav.shape[0] > 1:  # mix to mono
            wav = wav.mean(dim=0, keepdim=True)
        wav24k = F.resample(wav, sr, 24000)
        # sound = AudioSegment.from_file(wav_file)
        # sound = sound.set_frame_rate(24000)
        # sound = sound.set_channels(1)
        # samples = np.frombuffer(sound.raw_data, np.int16).flatten().astype(np.float32) / 32768.0
        # wav24k = torch.from_numpy(samples).unsqueeze(0)
        wave = wav24k.to(device)
    except Exception as e:
        print(e)
        print(f"Error with {wav_file}")
        return
    mel = mel_extractor(wave)
    torch.save(mel.cpu().detach(), outfile)