import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio
from pypinyin import Style, lazy_pinyin
import math
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.utils.infer_utils import load_model
import json
import os

def padding_to_8(x):
    l = x.shape[-1]
    l = (math.floor(l / 8) + 1) * 8
    x = torch.nn.functional.pad(x, (0, l-x.shape[-1]))
    return x
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

def padding_to_8(x):
    l = x.shape[-1]
    l = (math.floor(l / 8) + 1) * 8
    x = torch.nn.functional.pad(x, (0, l-x.shape[-1]))
    return x
class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.jsonl_path = opt['dataset']['path']
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)
        self.tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath, text = audiopath_and_text['path'], audiopath_and_text['text']
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = self.tok.encode(text)
        text_tokens = LongTensor(text)
        try:
            mel_path = audiopath + '.mel.pth'
            mel_raw = torch.load(mel_path)[0]

            quant_path = audiopath + '.melvq.pth'
            mel_codes = LongTensor(torch.load(quant_path)[0])
        except:
            return None

        # Define the number of frames for the random crop (adjust as needed)
        crop_frames = random.randint(int(mel_raw.shape[1] // 4), int(mel_raw.shape[1] // 4 * 3))

        # Ensure the crop doesn't exceed the length of the original audio
        max_start_frame = mel_raw.shape[1] - crop_frames
        start_frame = random.randint(0, max_start_frame)

        # Perform the random crop
        mel_refer = mel_raw[:, start_frame: start_frame + crop_frames]
        mel_refer = padding_to_8(mel_refer)
        # split = random.randint(int(mel_raw.shape[1]//3), int(mel_raw.shape[1]//3*2))
        # if random.random()>0.5:
        #     mel_refer = mel_raw[:,split:]
        # else:
        #     mel_refer = mel_raw[:,:split]
        # if mel_refer.shape[1]>200:
        #     mel_refer = mel_refer[:,:200]
        #text_token mel_codes 

        if mel_raw.shape[1]>400:
            mel_raw = mel_raw[:,:400]
            mel_codes = mel_codes[:100]
        if mel_codes.shape[-1]%2==1:
            mel_codes = mel_codes[:-1]
            mel_raw = mel_raw[:,:-4]
        return text_tokens, mel_codes, mel_raw, mel_refer

    def __len__(self):
        return len(self.audiopaths_and_text)


class DiffusionCollater():

    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch)==0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)
        mel_code_lens = [len(x[1]) for x in batch]
        max_mel_code_len = max(mel_code_lens)
        mel_lens = [x[2].shape[1] for x in batch]
        max_mel_len = max(mel_lens)
        mel_refer_lens = [x[3].shape[1] for x in batch]
        max_mel_refer_len = max(mel_refer_lens)
        texts = []
        mel_codes = []
        mels = []
        mel_refers = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text_token, mel_code, mel, mel_refer = b
            texts.append(F.pad(text_token,(0,max_text_len-len(text_token)), value=0))
            mel_codes.append(F.pad(mel_code,(0,max_mel_code_len-len(mel_code)), value=0))
            mels.append(F.pad(mel,(0, max_mel_len-mel.shape[1]), value=0))
            mel_refers.append(F.pad(mel_refer,(0, max_mel_refer_len-mel_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel_code = torch.stack(mel_codes)
        padded_mel = torch.stack(mels)
        padded_mel_refer = torch.stack(mel_refers)
        return {
            'padded_text': padded_text,
            'padded_mel_code': padded_mel_code,
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
    cfg = json.load(open('ttts/diffusion/config.json'))
    ds = DiffusionDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=DiffusionCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
