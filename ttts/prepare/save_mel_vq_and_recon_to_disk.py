import functools
import multiprocessing
import os

import torch
import torchaudio
from tqdm import tqdm
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.utils import find_audio_files, get_paths_with_cache
from ttts.utils.infer_utils import load_model

def process_one(paths):
    vqvae = load_model('vqvae', model_path, 'ttts/vqvae/config.json', 'cuda')
    mel_extractor = MelSpectrogramFeatures().cuda()
    with torch.no_grad():
        for path in tqdm(paths):
            try:
                audio,sr = torchaudio.load(path)
            except Exception as e:
                print(path)
                print(e)
                continue
            if audio.shape[0]>1:
                audio = audio[0].unsqueezze(0)
            audio = torchaudio.transforms.Resample(sr,24000)(audio).cuda()
            audio_length = audio.shape[1]
            # pad_to = int(audio_length/256/8+1)*256*8
            # audio = torch.nn.functional.pad(audio, (0, pad_to-audio_length)) 
            mel = mel_extractor(audio)
            code = vqvae.get_codebook_indices(mel)
            # code = code[:,:int(audio_length//1024)+4]
            outp = path+'.melvq.pth'
            mel_recon, _ = vqvae.infer(mel)
            mel_recon_outp = path+'.melrecon.pth'
            assert abs(mel.shape[2]-mel_recon.shape[2])<=3
            mel_len = min(mel.shape[2],mel_recon.shape[2])
            mel, mel_recon = mel[:,:,:mel_len], mel_recon[:,:,:mel_len]
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            torch.save(code.tolist(), outp)
            torch.save(mel_recon.detach().cpu(), mel_recon_outp)
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