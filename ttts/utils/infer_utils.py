from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.diffusion.model import DiffusionTts
from ttts.gpt.model import UnifiedVoice
from ttts.classifier.model import AudioMiniEncoderWithClassifierHead
import json
import torch

def load_model(model_name, model_path, config_path, device):
    config = json.load(open(config_path))
    if model_name=='vqvae':
        model = DiscreteVAE(**config['vqvae'])
        sd = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
    elif model_name=='gpt':
        model = UnifiedVoice(**config['gpt'])
        gpt = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(gpt, strict=True)
        model = model.to(device)
    elif model_name=='diffusion':
        model = DiffusionTts(**config['diffusion'])
        diffusion = torch.load(model_path, map_location=device)
        model.load_state_dict(diffusion, strict=True)
    elif model_name == 'classifier':
        model = AudioMiniEncoderWithClassifierHead(**config['classifier'])
        classifier = torch.load(model_path, map_location=device)
        model.load_state_dict(classifier, strict=True)
    # elif model_name=='clvp':

    
    return model.eval()