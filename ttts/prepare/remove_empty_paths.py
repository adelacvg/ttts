import json
import os
from tqdm import tqdm
def remove_nonexistent_files(jsonl_file):
    # 读取jsonl文件
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()

    # 解析每一行的JSON数据
    json_data_list = [json.loads(line.strip()) for line in lines]

    # 删除不存在的文件
    valid_json_data_list = []
    for json_data in tqdm(json_data_list):
        file_path = json_data.get("path", "")
        if os.path.exists(file_path):
            # print(f"文件存在: {file_path}")
            valid_json_data_list.append(json_data)
        else:
            print(f"删除不存在的文件: {file_path}")

    # 将更新后的JSON数据重新写入jsonl文件
    with open(jsonl_file, 'w') as f:
        for valid_json_data in valid_json_data_list:
            f.write(json.dumps(valid_json_data, ensure_ascii=False) + '\n')

# 替换为你的jsonl文件路径
jsonl_file_path = "/home/hyc/tortoise_plus_zh/ttts/datasets/filtered_paths.jsonl"

# 执行删除不存在文件的脚本并保存更新后的数据
remove_nonexistent_files(jsonl_file_path)