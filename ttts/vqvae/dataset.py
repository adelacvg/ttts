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
        self.paths = [os.path.join(pre,d['path'])+'.mel.pth' for d in paths]
        self.pad_to = opt['dataset']['pad_to_samples']
        self.squeeze = opt['dataset']['should_squeeze']

    def __getitem__(self, index):
        try:
            mel = torch.load(self.paths[index])
        except:
            mel = torch.zeros(1,100,self.pad_to)
        if mel.shape[-1] >= self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to+1, (1,))
            mel = mel[:, :, start:start+self.pad_to]
            mask = torch.zeros_like(mel)
        else:
            mask = torch.zeros_like(mel)
            padding_needed = self.pad_to - mel.shape[-1]
            mel = F.pad(mel, (0,padding_needed))
            mask = F.pad(mask, (0,padding_needed), value=1)
        assert mel.shape[-1] == self.pad_to
        if self.squeeze:
            mel = mel.squeeze()

        return mel

    def __len__(self):
        return len(self.paths)


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
    cfg = json.load(open('vqvae/config.json'))
    ds = PreprocessedMelDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'])
    i = 0
    for b in tqdm(dl):
        #pass
        torchvision.utils.save_image((b['mel']+1)/2, f'{i}.png')
        i += 1
        if i > 20:
            break