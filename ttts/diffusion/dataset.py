import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio
from pypinyin import Style, lazy_pinyin

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import json
import os

def read_jsonl(path):
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def write_jsonl(path, all_paths):
    with open(path,'w', encoding='utf-8') as file:
        for item in all_paths:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.jsonl_path = opt['dataset']['path']
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)

    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath, text = audiopath_and_text['path'], audiopath_and_text['text']
        # Fetch quantized MELs
        mel_recon_path = audiopath + '.melrecon.pth'
        mel_recon_raw = torch.load(mel_recon_path)[0]

        mel_path = audiopath + '.mel.pth'
        mel_raw = torch.load(mel_path)[0]
        split = random.randint(int(mel_raw.shape[1]//3), int(mel_raw.shape[1]//3*2))

        if random.random()>0.5:
            mel_recon = mel_recon_raw[:,:split]
            mel = mel_raw[:,:split]
            mel_refer = mel_raw[:,split:]
        else:
            mel_recon = mel_recon_raw[:,split:]
            mel = mel_raw[:,split:]
            mel_refer = mel_raw[:,:split]
        if mel.shape[1]>200:
            mel_recon = mel_recon[:,:200]
            mel = mel[:,:200]
        if mel_refer.shape[1]>100:
            mel_refer = mel_refer[:,:100]
        return mel_recon, mel, mel_refer

    def __len__(self):
        return len(self.audiopaths_and_text)


class DiffusionCollater():

    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch)==0:
            return None
        mel_recon_lens = [x[0].shape[1] for x in batch]
        max_mel_recon_len = max(mel_recon_lens)
        mel_lens = [x[1].shape[1] for x in batch]
        max_mel_len = max(mel_lens)
        mel_refer_lens = [x[2].shape[1] for x in batch]
        max_mel_refer_len = max(mel_refer_lens)
        mel_recons = []
        mels = []
        mel_refers = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            mel_recon, mel, mel_refer = b
            mel_recons.append(F.pad(mel_recon, (0, max_mel_recon_len-mel_recon.shape[1]), value=0))
            mels.append(F.pad(mel,(0, max_mel_len-mel.shape[1]), value=0))
            mel_refers.append(F.pad(mel_refer,(0, max_mel_refer_len-mel_refer.shape[1]), value=0))

        padded_mel_recon = torch.stack(mel_recons)
        padded_mel = torch.stack(mels)
        padded_mel_refer = torch.stack(mel_refers)
        return {
            'padded_mel_recon': padded_mel_recon,
            'padded_mel': padded_mel,
            'mel_lengths': LongTensor(mel_lens),
            'padded_mel_refer':padded_mel_refer,
            'mel_refer_lengths':LongTensor(mel_refer_lens)
        }


if __name__ == '__main__':
    params = {
        'mode': 'gpt_tts',
        'path': 'E:\\audio\\LJSpeech-1.1\\ljs_audio_text_train_filelist.txt',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'mel_vocab_size': 512,
    }
    cfg = json.load(open('diffusion/config.json'))
    ds = DiffusionDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=DiffusionCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
