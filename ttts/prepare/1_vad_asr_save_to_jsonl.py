from pathlib import Path
import torch
import os
from tqdm import tqdm
from ttts.utils.utils import get_paths_with_cache

os.environ["MODELSCOPE_CACHE"] = "./"


def phase1_vad_and_sample(file_paths, out_path, max_workers):
    paths = [[file_path, out_path] for file_path in file_paths]
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_file_vad, paths), total=len(file_paths), desc="VAD"))
    results = [result for result in results if result is not None]

def phase2_filter_and_transcript_and_to_jsonl(file_paths, out_path, max_workers):
    paths = [[file_path, out_path] for file_path in file_paths]
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_file_asr, paths), total=len(file_paths), desc="ASR"))
    results = [result for result in results if result is not None]


if __name__ == '__main__':
    # phase 1
    from ttts.prepare.vad_process import process_file_vad
    print("---------------phase1-----------------")
    # files = get_paths_with_cache('ttts/datasets/raw_datasets')
    # out_path = 'ttts/datasets/cliped_datasets_min1s'
    # Path(out_path).mkdir(exist_ok = True, parents=True)
    # phase1_vad_and_sample(files, out_path, 8)

    # phase 2 
    from ttts.prepare.asr_process import process_file_asr
    print("---------------phase2-----------------")
    files = get_paths_with_cache('ttts/datasets/hq_dataset')
    out_path = 'ttts/datasets/hq_data.jsonl'
    phase2_filter_and_transcript_and_to_jsonl(files, out_path, 8)



