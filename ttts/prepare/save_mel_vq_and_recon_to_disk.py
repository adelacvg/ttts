import os

import torch
import torchaudio
from tqdm import tqdm
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.utils import find_audio_files, load_model, get_paths_with_cache


if __name__ == '__main__':
    model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/2023-10-22-18-02-14/model-34.pt'
    vqvae = load_model('vqvae', model_path, 'vqvae/config.json', 'cuda')
    mel_extractor = MelSpectrogramFeatures().cuda()
    paths = get_paths_with_cache('datasets/cliped_datasets','datasets/wav_clip_path.cache')
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
            code = code[:,:int(audio_length//1024)+4]
            outp = path+'.melvq.pth'
            mel_recon, _ = vqvae.infer(mel)
            mel_recon_outp = path+'.melrecon.pth'
            assert abs(mel.shape[2]-mel_recon.shape[2])<=3
            mel_len = min(mel.shape[2],mel_recon.shape[2])
            mel, mel_recon = mel[:,:,:mel_len], mel_recon[:,:,:mel_len]
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            torch.save(code.tolist(), outp)
            torch.save(mel_recon.detach().cpu(), mel_recon_outp)