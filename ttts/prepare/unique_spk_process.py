import os
import random
import shutil
from tqdm import tqdm
from modelscope.pipelines import pipeline
import json

sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

# 定义文件夹路径
def process_unique_spk(paths):
    out_path, folder_path = paths
    # 获取文件夹下所有音频文件的路径
    audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]

    # 确保文件夹中有音频文件
    if len(audio_files) == 0:
        print("文件夹中没有音频文件。")
    elif len(audio_files) == 1:
        # 如果文件夹中只有一个文件，删除整个文件夹
        print(f"删除文件夹: {folder_path}")
        # shutil.rmtree(folder_path)  # 删除整个文件夹
    else:
        # 随机选择一个音频文件作为参考文件
        reference_file = random.choice(audio_files[1:])
        
        # 判断每个文件是否属于相同的说话人，如果不是则删除
        for audio_file in audio_files:
            # 验证文件是否属于相同说话人
            result = sv_pipeline([reference_file, audio_file])
            # 如果验证得分低于阈值，说明不是同一个说话人，将文件删除
            if result['text']=='no':
                print(f"删除文件: {audio_file}")
                with open(out_path, 'a', encoding='utf-8') as file:
                    json.dump({'path':audio_file}, file, ensure_ascii=False)
                    file.write('\n')
                # os.remove(audio_file)  # 删除单个文件root_path = '/home/hyc/tortoise_plus_zh/ttts/datasets/cliped_datasets_min1s'
