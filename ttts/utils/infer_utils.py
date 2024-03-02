from ttts.vqvae.dvae import DiscreteVAE
from ttts.diffusion.model import DiffusionTts
from ttts.gpt.model import UnifiedVoice
from ttts.classifier.model import AudioMiniEncoderWithClassifierHead
from omegaconf import OmegaConf
from ttts.diffusion.aa_model import AA_diffusion
import json
import torch
import os

def load_model(model_name, model_path, config_path, device):
    config_path = os.path.expanduser(config_path)
    model_path = os.path.expanduser(model_path)
    if config_path.endswith('.json'):
        config = json.load(open(config_path))
    else:
        config = OmegaConf.load(config_path)
    if model_name=='vqvae':
        model = RVQ1(**self.cfg['vqvae'])
        sd = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
    elif model_name=='gpt':
        model = UnifiedVoice(**config['gpt'])
        gpt = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(gpt, strict=True)
        model = model.to(device)
    elif model_name=='diffusion':
        # model = DiffusionTts(**config['diffusion'])
        model = AA_diffusion(config)
        diffusion = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(diffusion, strict=True)
        model = model.to(device)
    elif model_name == 'classifier':
        model = AudioMiniEncoderWithClassifierHead(**config['classifier'])
        classifier = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(classifier, strict=True)
        model = model.to(device)
    # elif model_name=='clvp':

    
    return model.eval()