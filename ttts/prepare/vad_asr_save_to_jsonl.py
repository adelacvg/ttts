import functools
import multiprocessing
from multiprocessing.pool import ThreadPool
from pathlib import Path
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio
import torch
import os
from tqdm import tqdm
from pydub.silence import split_on_silence
from pydub import  AudioSegment
from ttts.utils.utils import get_paths_with_cache
from pydub.exceptions import CouldntDecodeError
def phase1_vad_and_sample(audio_path,out_path):
    min_duration=2
    max_duration=20
    try:
        audio = AudioSegment.from_file(audio_path)
    except CouldntDecodeError as e:
        print(e)
        print(audio_path)
        return
    segments = split_on_silence(audio, min_silence_len=600, silence_thresh=-40, seek_step=100, keep_silence=50)
    out_prefix = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(os.path.join(out_path, out_prefix),exist_ok=True)

    for i in range(0,len(segments)):
        if segments[i].duration_seconds<min_duration or segments[i].duration_seconds > max_duration:
            continue
        if os.path.exists(os.path.join(out_path, out_prefix, str(i)+'.wav')):
            return
        clip_path = os.path.join(out_path, out_prefix,str(i)+'.wav')
        segments[i].export(clip_path, format='wav')

def phase2_filter_and_transcript_and_to_jsonl(clip_paths):
    all_paths = []
    asr_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    )
    for clip_path in tqdm(clip_paths):
        try:
            rec_result = asr_pipeline(audio_in = clip_path)
        except Exception as e:
            print(e)
            return
        text = rec_result['text']
        all_paths.append({'text':text,'path':clip_path})
    return all_paths

if __name__ == '__main__':
    # phase 1
    print("---------------phase1-----------------")
    files = get_paths_with_cache('ttts/datasets/raw_datasets', 'ttts/datasets/wav_paths.cache')
    cliped_path = 'ttts/datasets/cliped_datasets'
    Path(cliped_path).mkdir(exist_ok = True, parents=True)
    num_threads = 8
    with ThreadPool(num_threads) as pool:
        list(tqdm(pool.imap(functools.partial(phase1_vad_and_sample, out_path=cliped_path), files), total=len(files)))

    # phase 2 
    print("---------------phase2-----------------")
    files = get_paths_with_cache('ttts/datasets/cliped_datasets', 'ttts/datasets/clip_paths.cache')
    processed_path = 'ttts/datasets/processed_datasets'
    all_data_path = 'ttts/datasets/all_data.jsonl'
    Path(processed_path).mkdir(exist_ok = True, parents=True)
    num_threads = 6 
    funcs = []
    for i in range(num_threads):
        funcs.append(functools.partial(phase2_filter_and_transcript_and_to_jsonl, files[i::num_threads]))
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(func) for func in funcs]
        pool.close()
        pool.join()
    all_paths = sum([result.get() for result in results], [])
    with open(all_data_path, 'w', encoding='utf-8') as file:
        for item in all_paths:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')



