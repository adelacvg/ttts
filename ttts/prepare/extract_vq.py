import torchaudio
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.infer_utils import load_model
import torch
from tqdm import tqdm
import os

model_path = '~/tortoise_plus_zh/ttts/vqvae/logs/2023-11-24-01-21-25/model-30.pt'
vqvae = load_model('vqvae', model_path, 'ttts/vqvae/config.json', 'cuda')
mel_extractor = MelSpectrogramFeatures().cuda()
def process_vq(path):
    try:
        audio,sr = torchaudio.load(path)
    except Exception as e:
        print(path)
        print(e)
        return
    if audio.shape[0]>1:
        audio = audio[0].unsqueeze(0)
    if sr!=24000:
        audio = torchaudio.transforms.Resample(sr,24000)(audio).cuda()
    else:
        audio = audio.cuda()
    mel = mel_extractor(audio)
    with torch.no_grad():
        code = vqvae.get_codebook_indices(mel)
        outp = path+'.melvq.pth'
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        torch.save(code.tolist(), outp)
    return