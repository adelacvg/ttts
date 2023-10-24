import argparse
import os

import numpy
import torch
from spleeter.audio.adapter import AudioAdapter
import torchaudio
from tqdm import tqdm

# Uses pydub to process a directory of audio files, splitting them into clips at points where it detects a small amount
# of silence.
from ttts.utils.utils import find_audio_files
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    files = find_audio_files(args.path, ['.wav'])
    mel_extractor = MelSpectrogramFeatures()
    for e, wav_file in enumerate(tqdm(files)):
        if e < 0:
            continue
        print(f"Processing {wav_file}..")
        outfile = f'{wav_file}.mel.pth'
        if os.path.exists(outfile):
            continue

        try:
            wave, sample_rate = torchaudio.load(wav_file)
            if wave.size(0) > 1:  # mix to mono
                wave = wave[0].unsqueeze[0]
            if sample_rate!=24000:
                wave = torchaudio.functional.resample(wave, orig_freq=sample_rate, new_freq=24000)
        except:
            print(f"Error with {wav_file}")
            continue

        mel = mel_extractor(wave)
        torch.save(mel.cpu(), outfile)


if __name__ == '__main__':
    main()