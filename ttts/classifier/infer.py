from tqdm import tqdm
from ttts.utils.infer_utils import load_model
import json
import torch.nn.functional as F
import torch
import os
def read_jsonl(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def classify_audio_clip(clip, classifier):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    with torch.no_grad():
        results = F.softmax(classifier(clip), dim=-1)
    return results

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,paths):
        super().__init__()
        self.paths = paths
        self.pad_to=700
    def __getitem__(self,index):
        path = self.paths[index]
        try:
            mel = torch.load(path+'.mel.pth')
        except:
            mel = torch.zeros((1,100,self.pad_to))
        if mel.shape[-1] >= self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to+1, (1,))
            mel = mel[:, :, start:start+self.pad_to]
        else:
            padding_needed = self.pad_to - mel.shape[-1]
            mel = F.pad(mel, (0,padding_needed))
        mel = mel.squeeze(0)
        return mel,path
    def __len__(self):
        return len(self.paths)

if __name__=='__main__':
    model_path = '/home/hyc/tortoise_plus_zh/ttts/classifier/logs/2024-02-04-15-50-46/model-24.pt'
    config_path = '~/tortoise_plus_zh/ttts/classifier/config.json'
    device = 'cuda'
    classifier = load_model('classifier', model_path, config_path, device)
    jsonl_path = '~/tortoise_plus_zh/ttts/datasets/all_data.jsonl'
    
    audiopaths_and_text = read_jsonl(jsonl_path)
    audio_paths = [x['path'] for x in audiopaths_and_text]
    ds = MelDataset(audio_paths)
    dl = torch.utils.data.DataLoader(ds,batch_size=1024,num_workers=16)
    for _,batch in tqdm(enumerate(dl),total=len(dl)):
        mels, paths = batch
        mels = mels.to(device)
        label = classify_audio_clip(mels,classifier)
        for i in range(label.shape[0]):
            if label[i][0]<0.5:
                with open('ttts/classifier/noise_files.txt','a') as f:
                    # print(os.path.join(os.getcwd(),paths[i]))
                    f.write(os.path.join(os.getcwd(),paths[i])+'\n')
    