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
# cnhubert.cnhubert_base_path = '/home/hyc/tortoise_plus_zh/ttts/pretrained_models/chinese-hubert-base'
# hmodel=cnhubert.get_model().to(device)
hmodel = vc_utils.get_hubert_model().to(device)
def process_one(file_path):
    hubert_path = file_path + ".hubert.pt"
    # if os.path.exists(hubert_path):
    #     return
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav16k = F.resample(wav, sr, 16000)
    if wav16k.shape[-1]>24*16000:
        return
    wav16k = wav16k.to(device)
    # c = hmodel.model(wav16k)["last_hidden_state"].transpose(1,2)
    c = vc_utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k[0])
    torch.save(c.cpu(), hubert_path)

if __name__=='__main__':
    pass
