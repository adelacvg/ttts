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

class PreprocessedMelDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        super().__init__()
        clean = opt['dataset']['clean']
        noise = opt['dataset']['noise']
        self.labels = []
        self.paths = []
        with open(clean) as file:
            for line in file:
                line=line.strip()
                if line.endswith('.wav'):
                    self.paths.append(line.replace('.wav','.mel.path'))
                    self.labels += [0]
                else:
                    self.paths += [str(p) for p in Path(line).rglob("*.mel.pth")]
                    self.labels += [0 for _ in range(len(self.paths) - len(self.labels))]
        with open(noise) as file:
            for line in file:
                line=line.strip()
                if line.endswith('.wav'):
                    self.paths.append(line+'.mel.pth')
                    self.labels += [1]
                else:
                    self.paths += [str(p) for p in Path(line).rglob("*.mel.pth")]
                    self.labels += [0 for _ in range(len(self.paths) - len(self.labels))]
        
        self.pad_to = opt['dataset']['pad_to_samples']
        self.squeeze = opt['dataset']['should_squeeze']

    def __getitem__(self, index):
        mel = torch.load(self.paths[index])
        if mel.shape[-1] >= self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to+1, (1,))
            mel = mel[:, :, start:start+self.pad_to]
        else:
            padding_needed = self.pad_to - mel.shape[-1]
            mel = F.pad(mel, (0,padding_needed))
        assert mel.shape[-1] == self.pad_to
        if self.squeeze:
            mel = mel.squeeze()
        label = self.labels[index]
        return mel, label

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    cfg = json.load(open('ttts/classifier/config.json'))
    ds = PreprocessedMelDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'])
    i = 0
    for _, b in tqdm(enumerate(dl)):
        #pass
        torchvision.utils.save_image((b['mel']+1)/2, f'{i}.png')
        i += 1
        if i > 20:
            break