import os
import shutil
from tqdm import tqdm

def delete_folders_with_single_wav(folder_path):
    contents = os.listdir(folder_path)

    wav_files = [content for content in contents if content.endswith('.wav') and os.path.isfile(os.path.join(folder_path, content))]

    if len(wav_files) == 1:
        # 如果文件夹只包含一个.wav文件，则删除整个文件夹
        print(f"删除只包含一个.wav文件的文件夹: {folder_path}")
        shutil.rmtree(folder_path)
    else:
        for content in contents:
            content_path = os.path.join(folder_path, content)

            if os.path.isdir(content_path):
                delete_folders_with_single_wav(content_path)

folder_to_delete = '/home/hyc/tortoise_plus_zh/ttts/datasets/cliped_datasets'

for folder_path in tqdm(os.listdir(folder_to_delete)):
    sub_dir_path = os.path.join(folder_to_delete, folder_path)
    delete_folders_with_single_wav(sub_dir_path)