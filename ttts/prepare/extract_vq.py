import torchaudio
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.infer_utils import load_model
import torch
from tqdm import tqdm
import os

model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/2024-03-02-14-43-14/model-42.pt'
vqvae = load_model('vqvae', model_path, 'ttts/vqvae/config.json', 'cuda')
def process_vq(path):
    cvec_path = path + ".mel.pth"
    try:
        cvec = torch.load(cvec_path).cuda()
    except Exception as e:
        print(path)
        print(e)
        return
    with torch.no_grad():
        code = vqvae.extract_code(cvec).squeeze(0).squeeze(0)
        outp = path+'.vq.pth'
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        torch.save(code.tolist(), outp)
    return