import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch
from pydub import  AudioSegment
from ttts.utils import vc_utils
import numpy as np
from ttts.utils import cnhubert
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
cnhubert.cnhubert_base_path = '/home/hyc/tortoise_plus_zh/ttts/pretrained_models/chinese-hubert-base'
hmodel=cnhubert.get_model().to(device)
def process_one(file_path):
    hubert_path = file_path + ".hubert.pt"
    if os.path.exists(hubert_path):
        return
    # sound = AudioSegment.from_file(file_path)
    # sound = sound.set_frame_rate(16000)
    # sound = sound.set_channels(1)
    # samples = np.frombuffer(sound.raw_data, np.int16).flatten().astype(np.float32) / 32768.0
    # wav16k = torch.from_numpy(samples).unsqueeze(0)
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav16k = F.resample(wav, sr, 16000)
    if wav16k.shape[-1]>24*16000:
        return
    # wav2 = T.Resample(sr, 16000)(wav)
    # torchaudio.save('1.wav',wav1, 16000)
    # torchaudio.save('2.wav',wav2, 16000)
    # wav24k = T.Resample(sr, 24000)(wav)
    # wav24k_path = filename
    # if not os.path.exists(os.path.dirname(wav24k_path)):
    #     os.makedirs(os.path.dirname(wav24k_path))
    # torchaudio.save(wav24k_path, wav24k, 24000)
    wav16k = wav16k.to(device)
    c = hmodel.model(wav16k)["last_hidden_state"].transpose(1,2)
    torch.save(c.cpu(), hubert_path)

if __name__=='__main__':
    pass
