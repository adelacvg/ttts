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
from ttts.utils.infer_utils import load_model
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

class HifiGANDataset(torch.utils.data.Dataset):
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

        wav,sr = torchaudio.load(audiopath)
        if wav.shape[0]>1:
            wav = wav[0].unsqueeze(0)
        if sr!=24000:
            wav = torchaudio.transforms.Resample(sr,24000)(wav)

        quant_path = audiopath + '.melvq.pth'
        mel_codes = LongTensor(torch.load(quant_path)[0])

        split = random.randint(int(wav.shape[1]//3), int(wav.shape[1]//3*2))
        if random.random()>0.5:
            wav_refer = wav[:,split:]
        else:
            wav_refer = wav[:,:split]
        if wav_refer.shape[1]>(50*1024):
            wav_refer = wav_refer[:,:50*1024]
        #text_token mel_codes 

        if wav.shape[1]>102400:
            wav = wav[:,:102400]
            mel_codes = mel_codes[:100]

        return text_tokens, mel_codes, wav, wav_refer

    def __len__(self):
        return len(self.audiopaths_and_text)


class HiFiGANCollater():

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
        wav_lens = [x[2].shape[1] for x in batch]
        max_wav_len = max(wav_lens)
        wav_refer_lens = [x[3].shape[1] for x in batch]
        max_wav_refer_len = max(wav_refer_lens)
        texts = []
        mel_codes = []
        wavs = []
        wav_refers = []
        for b in batch:
            text_token, mel_code, wav, wav_refer = b
            texts.append(F.pad(text_token,(0,max_text_len-len(text_token)), value=0))
            mel_codes.append(F.pad(mel_code,(0,max_mel_code_len-len(mel_code)), value=0))
            wavs.append(F.pad(wav,(0, max_wav_len-wav.shape[1]), value=0))
            wav_refers.append(F.pad(wav_refer,(0, max_wav_refer_len-wav_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel_code = torch.stack(mel_codes)
        padded_wav = torch.stack(wavs)
        padded_wav_refer = torch.stack(wav_refers)
        return {
            'padded_text': padded_text,
            'padded_mel_code': padded_mel_code,
            'padded_wav': padded_wav,
            'padded_wav_refer':padded_wav_refer,
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
    cfg = json.load(open('ttts/hifigan/config.json'))
    ds = HifiGANDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=HiFiGANCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break