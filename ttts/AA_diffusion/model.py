import os
from pathlib import Path
import torch.nn.functional as F
from omegaconf import OmegaConf
import torch
import torchaudio
from tqdm.auto import tqdm
from dataset import DiffusionCollater, DiffusionDataset
from ldm.util import instantiate_from_config
from ttts.utils.utils import clean_checkpoints, plot_spectrogram_to_numpy, summarize
from accelerate import Accelerator
from vocos import Vocos
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
from ttts.diffusion.diffusion_util import denormalize_tacotron_mel
from ttts.utils.infer_utils import load_model
# import utils
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def cycle(dl):
    while True:
        for data in dl:
            yield data
def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
def get_state_dict(d):
    return d.get('state_dict', d)
def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict
class Trainer(object):
    def __init__(
        self,
        cfg_path = 'ttts/AA_diffusion/config.yaml',
    ):
        super().__init__()

        self.cfg = OmegaConf.load(cfg_path)
        self.accelerator = Accelerator()
        # model
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.gpt = load_model('gpt',self.cfg['dataset']['gpt_path'],'ttts/gpt/config.json','cuda')
        self.model = create_model(cfg_path)
        self.mel_length_compression = 4
        print("model params:", count_parameters(self.model))
        # sampling and training hyperparameters
        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']
        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        self.dataset = DiffusionDataset(self.cfg)
        dl = DataLoader(self.dataset, **self.cfg['dataloader'], collate_fn=DiffusionCollater())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        # optimizer

        self.opt = AdamW(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            # eval_ds = TestDataset(self.cfg['data']['val_files'], self.cfg, self.vocos)
            # self.eval_dl = DataLoader(eval_ds, batch_size = 1, shuffle = False, num_workers = self.cfg['train']['num_workers'])
            # self.eval_dl = iter(cycle(self.eval_dl))

            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
        }

        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(model_path, map_location=device)

        self.step = data['step']

        saved_state_dict = data['model']
        # saved_state_dict['unconditioned_embedding'] = torch.nn.Parameter(torch.randn(1,100,1))
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(saved_state_dict)

    def train(self):
        # print(1)
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                # with autograd.detect_anomaly():
                data = next(self.dl)
                data = {k: v.to(self.device) for k, v in data.items()}
                with torch.no_grad():
                    latent = self.gpt(data['padded_mel_refer'], data['padded_text'],
                        torch.tensor([data['padded_text'].shape[-1]], device=device), data['padded_mel_code'],
                        torch.tensor([data['padded_mel_code'].shape[-1]*self.mel_length_compression], device=device),
                        return_latent=True, clip_inputs=False).transpose(1,2)
                latent = F.interpolate(latent, size=data['padded_mel'].shape[-1], mode='nearest')
                data_ =  dict(jpg=data['padded_mel'], txt=data['padded_mel_refer'], hint=latent)
                with self.accelerator.autocast():
                    loss = accelerator.unwrap_model(self.model).training_step(data_)

                    model = accelerator.unwrap_model(self.model)
                    unused_params =[]
                    unused_params.extend(list(model.refer_model.out.parameters()))
                    unused_params.extend(list(model.cond_stage_model.visual.proj))
                    unused_params.extend(list(model.refer_model.output_blocks.parameters()))
                    unused_params.extend(list(model.refer_model.output_blocks.parameters()))
                    extraneous_addition = 0
                    for p in unused_params:
                        extraneous_addition = extraneous_addition + p.mean()
                    loss = loss + 0*extraneous_addition
                self.accelerator.backward(loss)

                grad_norm = get_grad_norm(self.model)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {loss:.4f}')

                accelerator.wait_for_everyone()
                if (self.step+1)%self.gradient_accumulate_every==0:
                    self.opt.step()
                    self.opt.zero_grad()

                accelerator.wait_for_everyone()
############################logging#############################################
                if accelerator.is_main_process and self.step % 100 == 0:
                    scalar_dict = {"loss/diff": loss, "loss/grad": grad_norm}

                    summarize(
                        writer=writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )

                if accelerator.is_main_process:

                    if self.step % self.save_and_sample_every == 0:
                        data = data
                        data = {k: v.to(self.device) for k, v in data.items()}
                        with torch.no_grad():
                            latent = self.gpt(data['padded_mel_refer'], data['padded_text'],
                                torch.tensor([data['padded_text'].shape[-1]], device=device), data['padded_mel_code'],
                                torch.tensor([data['padded_mel_code'].shape[-1]*self.mel_length_compression], device=device),
                                return_latent=True, clip_inputs=False).transpose(1,2)
                        latent = F.interpolate(latent, size=data['padded_mel'].shape[-1], mode='nearest')
                        data_ =  dict(jpg=data['padded_mel'], txt=data['padded_mel_refer'], hint=latent)

                        with torch.no_grad():
                            model = accelerator.unwrap_model(self.model)
                            model.eval()
                            milestone = self.step // self.save_and_sample_every
                            log = model.log_images(data_)
                            mel = log['samples_cfg'].detach().cpu()
                            mel = denormalize_tacotron_mel(mel)
                            model.train()
                        gen = self.vocos.decode(mel)
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), gen, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": gen,
                            })
                        image_dict = {
                                f"gt/mel": plot_spectrogram_to_numpy(data['padded_mel'][0, :, :].detach().unsqueeze(-1).cpu()),
                                f"gen/mel": plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                        }
                        summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            images=image_dict,
                            audio_sampling_rate=24000
                        )
                        keep_ckpts = self.cfg['train']['keep_ckpts']
                        if keep_ckpts > 0:
                            clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                        self.save(milestone)
                self.step += 1

                pbar.update(1)

        accelerator.print('training complete')
# example

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
