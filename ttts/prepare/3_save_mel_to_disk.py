import argparse
import os

import numpy
import torch
from spleeter.audio.adapter import AudioAdapter
import torchaudio
from tqdm import tqdm
from ttts.prepare.mel_extract import process_mel

# Uses pydub to process a directory of audio files, splitting them into clips at points where it detects a small amount
# of silence.
from ttts.utils.utils import find_audio_files


def process_mels(file_paths, max_workers):
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_mel, file_paths), total=len(file_paths), desc="Mel_extract"))
    # 过滤掉返回None的结果
    results = [result for result in results if result is not None]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',default='ttts/datasets/hq_dataset')
    args = parser.parse_args()
    files = find_audio_files(args.path, ['.wav'])
    process_mels(files,8)

if __name__ == '__main__':
    main()