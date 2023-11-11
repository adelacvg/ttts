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
import concurrent.futures

os.environ["MODELSCOPE_CACHE"] = "./"


def transcribe_worker(file_path: str, inference_pipeline):
    """
    Worker function for transcribing a segment of an audio file.
    """
    rec_result = inference_pipeline(audio_in=file_path)
    text = str(rec_result.get('text', '')).strip()
    text_without_spaces = text.replace(" ", "")
    # logger.critical(file_path)
    # logger.critical("text: " + text_without_spaces)
    return text_without_spaces


def phase1_vad_and_sample(args):
    audio_path,out_path = args
    min_duration=4
    max_duration=20
    out_prefix = os.path.splitext(os.path.basename(audio_path))[0]
    if os.path.exists(os.path.join(out_path, out_prefix)):
        return
    try:
        audio = AudioSegment.from_file(audio_path)
    except CouldntDecodeError as e:
        print(e)
        print(audio_path)
        return
    segments = split_on_silence(audio, min_silence_len=600, silence_thresh=-40, seek_step=100, keep_silence=50)

    for i in range(0,len(segments)):
        if segments[i].duration_seconds<min_duration or segments[i].duration_seconds > max_duration:
            continue
        if os.path.exists(os.path.join(out_path, out_prefix, str(i)+'.wav')):
            return
        clip_path = os.path.join(out_path, out_prefix,str(i)+'.wav')
        if not os.path.exists(os.path.join(out_path, out_prefix)):
            os.makedirs(os.path.join(out_path, out_prefix),exist_ok=True)
        segments[i].export(clip_path, format='wav')
    return 0

def phase2_filter_and_transcript_and_to_jsonl(file_paths, out_path, max_workers):
    workers = [
        pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            model_revision="v1.2.4"
        )
        for _ in range(max_workers)
    ]
    all_workers = workers * (len(file_paths) // max_workers) + workers[:len(file_paths) % max_workers]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(file_paths), max_workers), desc="转写进度: "):
            l, r = i, min(i + max_workers, len(file_paths))
            transcriptions = list(executor.map(
                transcribe_worker,
                file_paths[l:r],
                all_workers[l:r]
            ))
            for file_path, transcription in zip(file_paths[l:r], transcriptions):
                if transcription:
                    with open(out_path, 'a', encoding='utf-8') as file:
                        json.dump({'text':transcription,'path':file_path}, file, ensure_ascii=False)
                        file.write('\n')

def process_files(files, output_dir, fn):
    os.makedirs(output_dir, exist_ok=True)

    args = [(file, output_dir) for file in files]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(fn, args), total=len(files)))

    print("All files processed.")

if __name__ == '__main__':
    # phase 1
    print("---------------phase1-----------------")
    files = get_paths_with_cache('ttts/datasets/raw_datasets', 'ttts/datasets/wav_paths.cache')
    cliped_path = 'ttts/datasets/cliped_datasets'
    Path(cliped_path).mkdir(exist_ok = True, parents=True)
    process_files(files, cliped_path, phase1_vad_and_sample)

    # phase 2 
    print("---------------phase2-----------------")
    files = get_paths_with_cache('ttts/datasets/cliped_datasets', 'ttts/datasets/clip_paths.cache')
    processed_path = 'ttts/datasets/processed_datasets'
    out_file_path = 'ttts/datasets/all_data.jsonl'
    phase2_filter_and_transcript_and_to_jsonl(files, out_file_path, 8)



