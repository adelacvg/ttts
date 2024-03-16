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
from pypinyin import Style, lazy_pinyin
import librosa
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import wave
import logging
import subprocess
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

def get_duration(in_audio):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', in_audio], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

class VQGANDataset(torch.utils.data.Dataset):

    def __init__(self, hps):
        paths = read_jsonl(hps.dataset.path)
        pre = os.path.expanduser(hps.dataset.pre)
        self.paths = [os.path.join(pre,d['path']) for d in paths]
        self.texts = [d['text'] for d in paths]
        self.hop_length = hps.data.hop_length
        self.win_length = hps.data.win_length
        self.sampling_rate = hps.data.sampling_rate
        lengths = []
        filtered_paths = []
        filtered_texts = []
        for path,text in zip(self.paths,self.texts):
            size = os.path.getsize(path)
            duration = size / self.sampling_rate / 2
            if duration <54 and duration > 0.65:
                filtered_paths.append(path)
                filtered_texts.append(text)
                lengths.append(size // (2 * self.hop_length))
        self.paths = filtered_paths
        self.texts = filtered_texts
        self.filter_length = hps.data.filter_length
        self.tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
        self.lengths = lengths

    def __getitem__(self, index):
        text = self.texts[index]
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = self.tok.encode(text)
        text = torch.LongTensor(text)

        wav_path = self.paths[index]
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav32k = AuF.resample(wav, sr, 32000)
        if wav32k.shape[-1]<32*self.hop_length:
            print(wav_path)
            return None
        wav32k = wav32k[:,:int(self.hop_length * (wav32k.shape[-1]//self.hop_length))]
        return  wav32k, text

    def __len__(self):
        return len(self.paths)

class VQVAECollater():
    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(-1) for x in batch]),
            dim=0, descending=True)
        max_wav_len = max([x[0].size(1) for x in batch])
        max_text_len = max([x[1].size(0) for x in batch])

        wav_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)

        wav_padded.zero_()
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            text = row[1]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

        return {
            'wav': wav_padded.squeeze(1),
            'wav_lengths':wav_lengths,
            'text':text_padded,
            'text_lengths':text_lengths
        }

class BucketSampler(torch.utils.data.Sampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, shuffle=True):
        super().__init__()
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size
        self.shuffle = shuffle
        self.epoch = 0

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

