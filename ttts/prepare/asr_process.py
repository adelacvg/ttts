import json
from tqdm import tqdm
import os
import re
import pandas as pd
from tqdm import tqdm
import logging
import warnings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
warnings.filterwarnings('ignore')
logging.getLogger('modelscope').setLevel(logging.CRITICAL)

inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            model_revision="v1.2.4"
        )
def process_file_asr(paths):
    file_path, out_path = paths
    data = pd.DataFrame(columns=['file_path', 'text'])
    
    try:
        text = inference_pipeline(audio_in= file_path)['text']
        if len(text) >= 5:
            my_re = re.compile(r'[A-Za-z]', re.S)
            res = re.findall(my_re, text)
            if len(res): 
                #不符合就删除，否则后面也会生成bert文件
                pass
                # os.remove(os.path.join(input_directory, file))
            else:
                # 将数据添加到DataFrame中
                print(f'{file_path} ASR结果：{text}')
                with open(out_path, 'a', encoding='utf-8') as file:
                    json.dump({'text':text,'path':file_path}, file, ensure_ascii=False)
                    file.write('\n')
                return file_path, text
                # data=data.append({'file_path': os.path.join(input_directory, file),  'text': text}, ignore_index=True)
        else:
            pass
            # os.remove(os.path.join(input_directory, file))
        # print(f'{file_path} ASR结果：{text}')
    except Exception :
        print(f"ASR异常，错误样本:{file_path}")
        return None