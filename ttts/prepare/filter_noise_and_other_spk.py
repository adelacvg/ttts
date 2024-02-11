import json
import os
from tqdm import tqdm

def read_jsonl(path):
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def read_txt(path):
    with open(path, 'r') as f:
        txt_str = f.read()
    data_list = []
    for line in txt_str.splitlines():
        data_list.append(line)
    return data_list

all_paths = read_jsonl('ttts/datasets/all_data.jsonl')
noise_paths = set(read_txt('ttts/classifier/noise_files.txt'))
unique_spk_paths = read_jsonl('/home/hyc/tortoise_plus_zh/ttts/datasets/unique_deleted.jsonl')
unique_spk_paths = set([json['path'] for json in unique_spk_paths])
out_path = 'ttts/datasets/filtered_paths.jsonl'
for data in tqdm(all_paths):
    abs_path = os.path.expanduser(os.path.join('~/tortoise_plus_zh',data['path']))
    if abs_path not in noise_paths:
        with open(out_path, 'a', encoding='utf-8') as file:
            json.dump({'text':data['text'],'path':data['path']}, file, ensure_ascii=False)
            file.write('\n')