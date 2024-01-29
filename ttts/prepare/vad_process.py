from pydub.silence import split_on_silence
from pydub import  AudioSegment
from pydub.exceptions import CouldntDecodeError
import os
# out_path = 'ttts/datasets/cliped_datasets'
def process_file_vad(paths):
    audio_path,out_path = paths
    min_duration=1
    max_duration=20
    out_prefix = os.path.splitext(os.path.basename(audio_path))[0]
    if os.path.exists(os.path.join(out_path, out_prefix)):
        return
    try:
        audio = AudioSegment.from_file(audio_path)
    except:
        print(audio_path)
        return 0
    segments = split_on_silence(audio, min_silence_len=500, silence_thresh=-40, seek_step=100, keep_silence=50)

    for i in range(0,len(segments)):
        if segments[i].duration_seconds<min_duration or segments[i].duration_seconds > max_duration:
            continue
        if os.path.exists(os.path.join(out_path, out_prefix, str(i)+'.wav')):
            return 0
        clip_path = os.path.join(out_path, out_prefix,str(i)+'.wav')
        if not os.path.exists(os.path.join(out_path, out_prefix)):
            os.makedirs(os.path.join(out_path, out_prefix),exist_ok=True)
        segments[i].export(clip_path, format='wav')
    return 0