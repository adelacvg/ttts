import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
import torchvision
from tqdm import tqdm
from ttts.utils import vc_utils

from ttts.classifier.infer import read_jsonl

class PreprocessedMelDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        # cache_path = opt['dataset']['cache_path']  # Will fail when multiple paths specified, must be specified in this case.
        # if os.path.exists(cache_path):
        #     self.paths = torch.load(cache_path)
        # else:
        #     print("Building cache..")
        #     path = Path(path)
        #     self.paths = [str(p) for p in path.rglob("*.mel.pth")]
        #     torch.save(self.paths, cache_path)
        paths = read_jsonl(opt['dataset']['path'])
        pre = os.path.expanduser(opt['dataset']['pre'])
        self.paths = [os.path.join(pre,d['path']) for d in paths]
        self.pad_to = opt['dataset']['pad_to_samples']
        self.squeeze = opt['dataset']['should_squeeze']
        # self.expand_times = 1.875#24k 256hop 93.75hz 16k 320hop 50hz  93.75/50=1.875

    def __getitem__(self, index):
        wav_path = self.paths[index]
        mel_path = wav_path+'.mel.pth'
        hubert_path = wav_path+'.hubert.pt'
        try:
            mel = torch.load(mel_path)
            hubert = torch.load(hubert_path).detach()
        except:
            return None
        hubert = vc_utils.repeat_expand_2d(hubert.squeeze(0), mel.shape[-1]).unsqueeze(0)
        assert hubert.shape[-1]==mel.shape[-1]
        if mel.shape[-1] > self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to+1, (1,))
            mel = mel[:, :, start:start+self.pad_to]
            hubert = hubert[:, :, start:start+self.pad_to]
        padding_needed = self.pad_to - mel.shape[-1]
        if padding_needed > 0:
            mel = F.pad(mel, (0,padding_needed))
            hubert = F.pad(hubert, (0,padding_needed))
        return mel, hubert
        # if hubert.shape[-1] > self.pad_to:
        #     start = torch.randint(0, hubert.shape[-1] - self.pad_to+1, (1,))
        #     end = int((start+self.pad_to)*self.expand_times)
        #     mel = mel[:, :, int(start*self.expand_times):end]
        #     hubert = hubert[:, :, start:start+self.pad_to]
        # padding_needed = self.pad_to - hubert.shape[-1]
        # if padding_needed > 0:
        #     hubert = F.pad(hubert, (0,padding_needed))
        # padding_needed = int(int(self.pad_to*self.expand_times) - mel.shape[-1])
        # if padding_needed > 0:
        #     mel = F.pad(mel, (0,padding_needed))
        # mel = mel[:,:,:int(self.pad_to*self.expand_times)]
        # assert hubert.shape[-1] == self.pad_to
        # assert mel.shape[-1] == int(self.pad_to * self.expand_times)
        # if self.squeeze:
        #     mel = mel.squeeze()
        # mel = mel.detach()
        # hubert = hubert.detach()
        # return mel,hubert

    def __len__(self):
        return len(self.paths)

class VQVAECollater():
    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        mels = [t[0] for t in batch]
        huberts = [t[1] for t in batch]
        mel = torch.stack(mels)
        hubert = torch.stack(huberts)
        return {
            'mel': mel,
            'hubert': hubert,
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