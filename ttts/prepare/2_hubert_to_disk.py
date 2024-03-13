from functools import partial
import math
import multiprocessing
import os
import argparse
from random import shuffle
import torchaudio
import torchaudio.transforms as T

import torch
from glob import glob
from tqdm import tqdm
from hubert_one import process_one
import logging
from ttts.classifier.infer import read_jsonl
from ttts.utils.utils import find_audio_files

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa
import numpy as np

def process_batch(file_paths, max_workers):
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_one, file_paths), total=len(file_paths), desc="ssl_extract"))
    # 过滤掉返回None的结果
    results = [result for result in results if result is not None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',default='ttts/datasets/44k_dataset_clips')
    args = parser.parse_args()
    paths = find_audio_files(args.path, ['.wav'])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--json_path',default='ttts/datasets/small_data.jsonl')
    # args = parser.parse_args()
    # paths = read_jsonl(args.json_path)
    # paths = [os.path.join('/home/hyc/tortoise_plus_zh',path['path']) for path in paths]
    num_threads = 4
    process_batch(paths, num_threads)