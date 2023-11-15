import functools
import multiprocessing
from multiprocessing.pool import ThreadPool
from pathlib import Path
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import os
from tqdm import tqdm
# from ttts.prepare import vad_process,asr_process
from ttts.prepare.vad_process import process_file_vad
from ttts.utils.utils import get_paths_with_cache

os.environ["MODELSCOPE_CACHE"] = "./"


def phase1_vad_and_sample(file_paths, max_workers):
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_file_vad, file_paths), total=len(file_paths), desc="VAD"))
    # 过滤掉返回None的结果
    results = [result for result in results if result is not None]

def phase2_filter_and_transcript_and_to_jsonl(file_paths, max_workers):
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_file_asr, file_paths), total=len(file_paths), desc="ASR"))
    # 过滤掉返回None的结果
    results = [result for result in results if result is not None]


if __name__ == '__main__':
    # phase 1
    print("---------------phase1-----------------")
    files = get_paths_with_cache('ttts/datasets/raw_datasets', 'ttts/datasets/wav_paths.cache')
    cliped_path = 'ttts/datasets/cliped_datasets'
    Path(cliped_path).mkdir(exist_ok = True, parents=True)
    # vad_process.out_path = cliped_path
    phase1_vad_and_sample(files, 8)

    # phase 2 

    from ttts.prepare.asr_process import process_file_asr
    print("---------------phase2-----------------")
    files = get_paths_with_cache('ttts/datasets/cliped_datasets', 'ttts/datasets/clip_paths.cache')
    processed_path = 'ttts/datasets/processed_datasets'
    out_file_path = 'ttts/datasets/all_data.jsonl'
    # asr_process.out_path = out_file_path
    phase2_filter_and_transcript_and_to_jsonl(files, 8)



