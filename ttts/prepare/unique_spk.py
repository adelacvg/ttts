import torch
from tqdm import tqdm
import os

root_path = '/home/hyc/tortoise_plus_zh/ttts/datasets/cliped_datasets_min1s'
out_path = 'ttts/datasets/unique_deleted.jsonl'
def do_unique_spk(file_paths, max_workers):
    paths = [[out_path, os.path.join(root_path, file_path)] for file_path in file_paths]
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_unique_spk, paths), total=len(file_paths), desc="UNISPK"))
    results = [result for result in results if result is not None]

if __name__ == '__main__':
    from unique_spk_process import process_unique_spk
    files = os.listdir(root_path)
    do_unique_spk(files, 16)


