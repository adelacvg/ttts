import torch
import torchaudio

# from data.audio.unsupervised_audio_dataset import load_audio
from ttts.script.do_to_files import do_to_files


def get_spec_mags(clip):
    stft = torch.stft(clip, n_fft=22000, hop_length=1024, return_complex=True)
    stft = stft[0, -2000:, :]
    return (stft.real ** 2 + stft.imag ** 2).sqrt()


def filter_no_hifreq_data(path, output_path):
    clip,sr = torch.load(path)
    if clip.shape[-1] < 22050:
        return
    stft = get_spec_mags(clip)
    if stft.mean() < .08:
        with open(output_path, 'a') as o:
            o.write(f'{path}\n')

if __name__ == '__main__':
    do_to_files(filter_no_hifreq_data)