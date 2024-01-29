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

def find_and_randomly_select_mel_files(wav_file):
    # 获取.wav文件所在文件夹路径
    wav_folder = os.path.dirname(wav_file)

    # 找到相同路径下的所有以.mel.pth结尾的文件
    mel_files = [f for f in os.listdir(wav_folder) if f.endswith('.mel.pth')]

    if mel_files:
        # 随机选择一个.mel.pth文件
        selected_mel_file = random.choice(mel_files)

        # 返回完整路径
        selected_mel_file_path = os.path.join(wav_folder, selected_mel_file)
        return selected_mel_file_path
    else:
        print("未找到以.mel.pth结尾的文件")
class GptTtsDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
        self.jsonl_path = opt['dataset']['path']
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)

    def __getitem__(self, index):
        try:
            # Fetch text and add start/stop tokens.
            audiopath_and_text = self.audiopaths_and_text[index]
            audiopath, text = audiopath_and_text['path'], audiopath_and_text['text']
            text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
            text = self.tok.encode(text)
            text = LongTensor(text)

            # Fetch quantized MELs
            quant_path = audiopath + '.melvq.pth'
            qmel = LongTensor(torch.load(quant_path)[0])

            mel_raw_path = audiopath + '.mel.pth'
            mel_raw = torch.load(mel_raw_path)[0]
            wav_length = mel_raw.shape[1]*256
            mel_path = find_and_randomly_select_mel_files(audiopath)
            mel = torch.load(mel_path)[0]
            split = random.randint(int(mel.shape[1]//3), int(mel.shape[1]//3*2))
            if random.random()>0.5:
                mel = mel[:,:split]
            else:
                mel = mel[:,split:]
        except Exception as e:
            print(e)
            return None

        #load wav
        # wav,sr = torchaudio.load(audiopath)
        # wav = torchaudio.transforms.Resample(sr,24000)(wav)
        if text.shape[0]>400 or qmel.shape[0]>800:
            return None

        return text, qmel, mel, wav_length

    def __len__(self):
        return len(self.audiopaths_and_text)


class GptTtsCollater():

    def __init__(self,cfg):
        self.cfg=cfg
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch)==0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)
        # max_text_len = self.cfg['gpt']['max_text_tokens']
        qmel_lens = [len(x[1]) for x in batch]
        max_qmel_len = max(qmel_lens)
        # max_qmel_len = self.cfg['gpt']['max_mel_tokens']
        raw_mel_lens = [x[2].shape[1] for x in batch]
        max_raw_mel_len = max(raw_mel_lens)
        wav_lens = [x[3] for x in batch]
        max_wav_len = max(wav_lens)
        texts = []
        qmels = []
        raw_mels = []
        wavs = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text, qmel, raw_mel, wav = b
            text = F.pad(text, (0, max_text_len-len(text)), value=0)
            texts.append(text)
            qmels.append(F.pad(qmel, (0, max_qmel_len-len(qmel)), value=0))
            raw_mels.append(F.pad(raw_mel,(0, max_raw_mel_len-raw_mel.shape[1]), value=0))

        padded_qmel = torch.stack(qmels)
        padded_raw_mel = torch.stack(raw_mels)
        padded_texts = torch.stack(texts)
        return {
            'padded_text': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'padded_qmel': padded_qmel,
            'qmel_lengths': LongTensor(qmel_lens),
            'padded_raw_mel': padded_raw_mel,
            'raw_mel_lengths': LongTensor(raw_mel_lens),
            'wav_lens': LongTensor(wav_lens)
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
    cfg = json.load(open('ttts/gpt/config.json'))
    ds = GptTtsDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=GptTtsCollater(cfg))
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
