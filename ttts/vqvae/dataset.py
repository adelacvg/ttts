import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AuF
import torch.utils.data
import torchaudio
import torchvision
from tqdm import tqdm
from ttts.utils import vc_utils
from ttts.utils.data_utils import spectrogram_torch
from ttts.classifier.infer import read_jsonl

class PreprocessedMelDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        paths = read_jsonl(opt['dataset']['path'])
        pre = os.path.expanduser(opt['dataset']['pre'])
        self.paths = [os.path.join(pre,d['path']) for d in paths]
        self.pad_to = opt['dataset']['pad_to_samples']
        self.squeeze = opt['dataset']['should_squeeze']
        self.win_length = opt['data']['win_length']
        self.hop_length = opt['data']['hop_length']
        self.filter_length = opt['data']['filter_length']

    def __getitem__(self, index):
        wav_path = self.paths[index]
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:  # mix to mono
            wav = wav.mean(dim=0, keepdim=True)
        wav32k = AuF.resample(wav, sr, 32000)
        spec = spectrogram_torch(wav32k, self.filter_length, self.hop_length,
                                 self.win_length, center=False)
        hubert_path = wav_path+'.hubert.pt'
        try:
            hubert = torch.load(hubert_path).detach()
        except:
            return None
        if (hubert.shape[-1] != spec.shape[-1]) and abs(hubert.shape[-1]-spec.shape[-1])==1:
            hubert = F.pad(hubert, (0, 1), mode="replicate")
        hubert = vc_utils.repeat_expand_2d(hubert.squeeze(0), spec.shape[-1]).unsqueeze(0)
        assert hubert.shape[-1]==spec.shape[-1]
        spec_raw = spec
        if spec.shape[-1] > self.pad_to:
            start = torch.randint(0, spec.shape[-1] - self.pad_to, (1,))
            spec = spec[:, :, start:start+self.pad_to]
            hubert = hubert[:, :, start:start+self.pad_to]
            wav = wav32k[:,(start*self.hop_length):((start+self.pad_to)*self.hop_length)]
        padding_needed = self.pad_to - spec.shape[-1]
        if padding_needed > 0:
            spec = F.pad(spec, (0,padding_needed))
            hubert = F.pad(hubert, (0,padding_needed))
            wav = F.pad(wav32k, (0,(self.pad_to*self.hop_length-wav32k.shape[-1])))
        if spec_raw.shape[-1]==self.pad_to:
            wav = wav32k[:,:self.pad_to*self.hop_length]
        try:
            assert wav.shape[-1]==self.pad_to*self.hop_length
        except:
            print(wav.shape)
        return spec, hubert, wav

    def __len__(self):
        return len(self.paths)

class VQVAECollater():
    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        specs = [t[0] for t in batch]
        huberts = [t[1] for t in batch]
        wav = [t[2] for t in batch]
        spec = torch.stack(specs).squeeze(1)
        hubert = torch.stack(huberts).squeeze(1)
        wav = torch.stack(wav).squeeze(1)
        return {
            'spec': spec,
            'hubert': hubert,
            'wav': wav
        }



if __name__ == '__main__':
    params = {
        'mode': 'preprocessed_mel',
        'path': 'Y:\\separated\\large_mel_cheaters',
        'cache_path': 'Y:\\separated\\large_mel_cheaters_win.pth',
        'pad_to_samples': 646,
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
    }
    cfg = json.load(open('ttts/vqvae/config.json'))
    ds = PreprocessedMelDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'])
    i = 0
    for b in tqdm(dl):
        #pass
        torchvision.utils.save_image((b['mel']+1)/2, f'{i}.png')
        i += 1
        if i > 20:
            break