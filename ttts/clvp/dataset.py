import os

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

class CLVPDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.tok = VoiceBpeTokenizer('gpt/gpt_tts_tokenizer.json')
        self.jsonl_path = opt['dataset']['path']
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)

    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath, text = audiopath_and_text['path'], audiopath_and_text['text']
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = self.tok.encode(text)
        text = LongTensor(text)

        # Fetch quantized MELs
        quant_path = audiopath + '.melvq.pth'
        qmel = LongTensor(torch.load(quant_path)[0])


        return text, qmel

    def __len__(self):
        return len(self.audiopaths_and_text)


class CLVPCollater():

    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch)==0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)
        qmel_lens = [len(x[1]) for x in batch]
        max_qmel_len = max(qmel_lens)
        texts = []
        qmels = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text, qmel = b
            text = F.pad(text, (0, max_text_len-len(text)), value=0)
            texts.append(text)
            qmels.append(F.pad(qmel, (0, max_qmel_len-len(qmel)), value=0))

        padded_qmel = torch.stack(qmels)
        padded_texts = torch.stack(texts)
        return {
            'padded_text': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'padded_qmel': padded_qmel,
            'qmel_lengths': LongTensor(qmel_lens),
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
    cfg = json.load(open('gpt/config.json'))
    ds = CLVPDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=CLVPCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
