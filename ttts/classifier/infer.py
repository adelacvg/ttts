from ttts.gpt.dataset import read_jsonl
from ttts.utils.infer_utils import load_model
import torch.nn.functional as F
def classify_audio_clip(clip, classifier):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

if __name__=='__main__':
    model_path = '~/tortoise_plus_zh/ttts/classifier/logs/2023-11-22-01-14-15/model-4.pt'
    config_path = '~/tortoise_plus_zh/ttts/classifier/config.json'
    device = 'cuda'
    classifier = load_model('classifier', model_path, config_path, device)
    jsonl_path = '~/tortoise_plus_zh/ttts/datasets/all_data.jsonl'
    audiopaths_and_text = read_jsonl(jsonl_path)
    