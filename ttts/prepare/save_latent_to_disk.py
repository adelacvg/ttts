
import functools
import multiprocessing
import os

import torch
import torchaudio
from tqdm import tqdm
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.utils import find_audio_files, get_paths_with_cache
from ttts.utils.infer_utils import load_model

def process_one(paths):
    gpt_path = '/home/hyc/tortoise_plus_zh/ttts/gpt/logs/2023-10-31-10-32-35/model-9.pt'
    gpt = load_model('gpt',gpt_path,'ttts/gpt/config.json','cuda')
    tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
    with torch.no_grad():
        for path in tqdm(paths):
            latent = gpt(mel_refer, text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), mel_codes,
                torch.tensor([mel_codes.shape[-1]*self.gpt.mel_length_compression], device=text_tokens.device),
                return_latent=True, clip_inputs=False)
            try:
                audio,sr = torchaudio.load(path)
            except Exception as e:
                print(path)
                print(e)
                continue
            if audio.shape[0]>1:
                audio = audio[0].unsqueezze(0)
            audio = torchaudio.transforms.Resample(sr,24000)(audio).cuda()
            mel = mel_extractor(audio)
            code = vqvae.get_codebook_indices(mel)
            outp = path+'.melvq.pth'
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            torch.save(code.tolist(), outp)
    return 0
    

if __name__ == '__main__':
    model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/2023-10-31-02-33-25/model-12.pt'
    paths = get_paths_with_cache('ttts/datasets/cliped_datasets','ttts/datasets/wav_clip_path.cache')
    num_threads = 8 
    funcs = []
    for i in range(num_threads):
        funcs.append(functools.partial(process_one, paths[i::num_threads]))
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(func) for func in funcs]
        pool.close()
        pool.join()
    all_paths = sum([result.get() for result in results], 0)