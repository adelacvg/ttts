import random

import torch
import torch.nn.functional as F
import torchaudio

from utils.util import opt_get, load_model_from_config, pad_or_truncate

# Base class for all other injectors.
class Injector(torch.nn.Module):
    def __init__(self, opt, env):
        super(Injector, self).__init__()
        self.opt = opt
        self.env = env
        if 'in' in opt.keys():
            self.input = opt['in']
        if 'out' in opt.keys():
            self.output = opt['out']

    # This should return a dict of new state variables.
    def forward(self, state):
        raise NotImplementedError
MEL_MIN = -11.512925148010254
TACOTRON_MEL_MAX = 2.3143386840820312
TORCH_MEL_MAX = 4.82  # FYI: this STILL isn't assertive enough...

def normalize_torch_mel(mel):
    return 2 * ((mel - MEL_MIN) / (TORCH_MEL_MAX - MEL_MIN)) - 1

def denormalize_torch_mel(norm_mel):
    return ((norm_mel+1)/2) * (TORCH_MEL_MAX - MEL_MIN) + MEL_MIN

def normalize_mel(mel):
    return 2 * ((mel - MEL_MIN) / (TACOTRON_MEL_MAX - MEL_MIN)) - 1

def denormalize_mel(norm_mel):
    return ((norm_mel+1)/2) * (TACOTRON_MEL_MAX - MEL_MIN) + MEL_MIN

class MelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from stft import TacotronSTFT
        # These are the default tacotron values for the MEL spectrogram.
        filter_length = opt_get(opt, ['filter_length'], 1024)
        hop_length = opt_get(opt, ['hop_length'], 256)
        win_length = opt_get(opt, ['win_length'], 1024)
        n_mel_channels = opt_get(opt, ['n_mel_channels'], 80)
        mel_fmin = opt_get(opt, ['mel_fmin'], 0)
        mel_fmax = opt_get(opt, ['mel_fmax'], 8000)
        sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        self.stft = TacotronSTFT(filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.do_normalization = opt_get(opt, ['do_normalization'], None)  # This is different from the TorchMelSpectrogramInjector. This just normalizes to the range [-1,1]

    def forward(self, state):
        inp = state[self.input]
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.stft = self.stft.to(inp.device)
        mel = self.stft.mel_spectrogram(inp)
        if self.do_normalization:
            mel = normalize_mel(mel)
        return {self.output: mel}


class TorchMelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = opt_get(opt, ['filter_length'], 1024)
        self.hop_length = opt_get(opt, ['hop_length'], 256)
        self.win_length = opt_get(opt, ['win_length'], 1024)
        self.n_mel_channels = opt_get(opt, ['n_mel_channels'], 80)
        self.mel_fmin = opt_get(opt, ['mel_fmin'], 0)
        self.mel_fmax = opt_get(opt, ['mel_fmax'], 8000)
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        norm = opt_get(opt, ['normalize'], False)
        self.true_norm = opt_get(opt, ['true_normalization'], False)
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=norm,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = opt_get(opt, ['mel_norm_file'], None)
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, state):
        with torch.no_grad():
            inp = state[self.input]
            if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
                inp = inp.squeeze(1)
            assert len(inp.shape) == 2
            self.mel_stft = self.mel_stft.to(inp.device)
            mel = self.mel_stft(inp)
            # Perform dynamic range compression
            mel = torch.log(torch.clamp(mel, min=1e-5))
            if self.mel_norms is not None:
                self.mel_norms = self.mel_norms.to(mel.device)
                mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
            if self.true_norm:
                mel = normalize_torch_mel(mel)
            return {self.output: mel}


class RandomAudioCropInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        if 'crop_size' in opt.keys():
            self.min_crop_sz = opt['crop_size']
            self.max_crop_sz = self.min_crop_sz
        else:
            self.min_crop_sz = opt['min_crop_size']
            self.max_crop_sz = opt['max_crop_size']
        self.lengths_key = opt['lengths_key']
        self.crop_start_key = opt['crop_start_key']
        self.min_buffer = opt_get(opt, ['min_buffer'], 0)
        self.rand_buffer_ptr=9999
        self.rand_buffer_sz=5000


    def forward(self, state):
        inp = state[self.input]
        if torch.distributed.get_world_size() > 1:
            # All processes should agree, otherwise all processes wait to process max_crop_sz (effectively). But agreeing too often
            # is expensive, so agree on a "chunk" at a time.
            if self.rand_buffer_ptr >= self.rand_buffer_sz:
                self.rand_buffer = torch.randint(self.min_crop_sz, self.max_crop_sz, (self.rand_buffer_sz,), dtype=torch.long, device=inp.device)
                torch.distributed.broadcast(self.rand_buffer, 0)
                self.rand_buffer_ptr = 0
            crop_sz = self.rand_buffer[self.rand_buffer_ptr]
            self.rand_buffer_ptr += 1
        else:
            crop_sz = random.randint(self.min_crop_sz, self.max_crop_sz)
        if self.lengths_key is not None:
            lens = state[self.lengths_key]
            len = torch.min(lens)
        else:
            len = inp.shape[-1]
            
        margin = len - crop_sz - self.min_buffer * 2
        if margin < 0:
            start = self.min_buffer
        else:
            start = random.randint(0, margin) + self.min_buffer

        res = {self.output: inp[:, :, start:start+crop_sz]}
        if self.crop_start_key is not None:
            res[self.crop_start_key] = start
        return res


class AudioClipInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.clip_size = opt['clip_size']
        self.ctc_codes = opt['ctc_codes_key']
        self.output_ctc = opt['ctc_out_key']

    def forward(self, state):
        inp = state[self.input]
        ctc = state[self.ctc_codes]
        len = inp.shape[-1]
        if len > self.clip_size:
            proportion_inp_remaining = self.clip_size/len
            inp = inp[:, :, :self.clip_size]
            ctc = ctc[:,:int(proportion_inp_remaining*ctc.shape[-1])]
        return {self.output: inp, self.output_ctc: ctc}


class AudioResampleInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.input_sr = opt['input_sample_rate']
        self.output_sr = opt['output_sample_rate']

    def forward(self, state):
        inp = state[self.input]
        return {self.output: torchaudio.functional.resample(inp, self.input_sr, self.output_sr)}


class DiscreteTokenInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        cfg = opt_get(opt, ['dvae_config'], "../experiments/train_diffusion_vocoder_22k_level.yml")
        dvae_name = opt_get(opt, ['dvae_name'], 'dvae')
        self.dvae = load_model_from_config(cfg, dvae_name, device=f'cuda:{env["device"]}').eval()

    def forward(self, state):
        inp = state[self.input]
        with torch.no_grad():
            self.dvae = self.dvae.to(inp.device)
            codes = self.dvae.get_codebook_indices(inp)
            return {self.output: codes}


class GptVoiceLatentInjector(Injector):
    """
    This injector does all the legwork to generate latents out of a UnifiedVoice model, including encoding all audio
    inputs into a MEL spectrogram and discretizing the inputs.
    """
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # For discrete tokenization.
        cfg = opt_get(opt, ['dvae_config'], "../experiments/train_diffusion_vocoder_22k_level.yml")
        dvae_name = opt_get(opt, ['dvae_name'], 'dvae')
        self.dvae = load_model_from_config(cfg, dvae_name).cuda().eval()
        # The unified_voice model.
        cfg = opt_get(opt, ['gpt_config'], "../experiments/train_gpt_tts_unified.yml")
        model_name = opt_get(opt, ['gpt_name'], 'gpt')
        pretrained_path = opt['gpt_path']
        self.gpt = load_model_from_config(cfg, model_name=model_name,
                                          also_load_savepoint=False, load_path=pretrained_path).cuda().eval()
        self.needs_move = True
        # Mel converter
        self.mel_inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'mel_norm_file': '../experiments/clips_mel_norms.pth'},{})
        # Aux input keys.
        self.conditioning_key = opt['conditioning_clip']
        self.text_input_key = opt['text']
        self.text_lengths_key = opt['text_lengths']
        self.input_lengths_key = opt['input_lengths']

    def to_mel(self, t):
        return self.mel_inj({'wav': t})['mel']

    def forward(self, state):
        with torch.no_grad():
            mel_inputs = self.to_mel(state[self.input])
            state_cond = pad_or_truncate(state[self.conditioning_key], 132300)
            mel_conds = []
            for k in range(state_cond.shape[1]):
                mel_conds.append(self.to_mel(state_cond[:, k]))
            mel_conds = torch.stack(mel_conds, dim=1)

            if self.needs_move:
                self.dvae = self.dvae.to(mel_inputs.device)
                self.gpt = self.gpt.to(mel_inputs.device)
            codes = self.dvae.get_codebook_indices(mel_inputs)
            latents = self.gpt(mel_conds, state[self.text_input_key],
                               state[self.text_lengths_key], codes, state[self.input_lengths_key],
                               text_first=True, raw_mels=None, return_attentions=False, return_latent=True)
            assert latents.shape[1] == codes.shape[1]
            return {self.output: latents}


class ReverseUnivnetInjector(Injector):
    """
    This injector specifically builds inputs and labels for a univnet detector.g
    """
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from scripts.audio.gen.speech_synthesis_utils import load_univnet_vocoder
        self.univnet = load_univnet_vocoder().cuda()
        self.mel_input_key = opt['mel']
        self.label_output_key = opt['labels']
        self.do_augmentations = opt_get(opt, ['do_aug'], True)

    def forward(self, state):
        with torch.no_grad():
            original_audio = state[self.input]
            mel = state[self.mel_input_key]
            decoded_mel = self.univnet.inference(mel)[:,:,:original_audio.shape[-1]]

            if self.do_augmentations:
                original_audio = original_audio + torch.rand_like(original_audio) * random.random() * .005
                decoded_mel = decoded_mel + torch.rand_like(decoded_mel) * random.random() * .005
                if(random.random() < .5):
                    original_audio = torchaudio.functional.resample(torchaudio.functional.resample(original_audio, 24000, 10000), 10000, 24000)
                if(random.random() < .5):
                    decoded_mel = torchaudio.functional.resample(torchaudio.functional.resample(decoded_mel, 24000, 10000), 10000, 24000)
                if(random.random() < .5):
                    original_audio = torchaudio.functional.resample(original_audio, 24000, 22000 + random.randint(0,2000))
                if(random.random() < .5):
                    decoded_mel = torchaudio.functional.resample(decoded_mel, 24000, 22000 + random.randint(0,2000))

                smallest_dim = min(original_audio.shape[-1], decoded_mel.shape[-1])
                original_audio = original_audio[:,:,:smallest_dim]
                decoded_mel = decoded_mel[:,:,:smallest_dim]

            labels = (torch.rand(mel.shape[0], 1, 1, device=mel.device) > .5)
            output = torch.where(labels, original_audio, decoded_mel)

            return {self.output: output, self.label_output_key: labels[:,0,0].long()}


class ConditioningLatentDistributionDivergenceInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        if 'gpt_config' in opt.keys():
            # The unified_voice model.
            cfg = opt_get(opt, ['gpt_config'], "../experiments/train_gpt_tts_unified.yml")
            model_name = opt_get(opt, ['gpt_name'], 'gpt')
            pretrained_path = opt['gpt_path']
            self.latent_producer = load_model_from_config(cfg, model_name=model_name,
                                                          also_load_savepoint=False, load_path=pretrained_path).eval()
            self.mel_inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'mel_norm_file': '../experiments/clips_mel_norms.pth'},{})
        else:
            from models.audio.tts.unet_diffusion_tts_flat import DiffusionTtsFlat
            self.latent_producer = DiffusionTtsFlat(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                          in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False,
                                          num_heads=16, layer_drop=0, unconditioned_percentage=0).eval()
            self.latent_producer.load_state_dict(torch.load(opt['diffusion_path']))
            self.mel_inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel', 'mel_fmax': 12000, 'sampling_rate': 24000, 'n_mel_channels': 100},{})
        self.needs_move = True
        # Aux input keys.
        self.conditioning_key = opt['conditioning_clip']
        # Output keys
        self.var_loss_key = opt['var_loss']

    def to_mel(self, t):
        return self.mel_inj({'wav': t})['mel']

    def forward(self, state):
        with torch.no_grad():
            state_preds = state[self.input]
            state_cond = pad_or_truncate(state[self.conditioning_key], 132300)
            mel_conds = []
            for k in range(state_cond.shape[1]):
                mel_conds.append(self.to_mel(state_cond[:, k]))
            mel_conds = torch.stack(mel_conds, dim=1)

            if self.needs_move:
                self.latent_producer = self.latent_producer.to(mel_conds.device)
            latents = self.latent_producer.get_conditioning_latent(mel_conds)

        sp_means, sp_vars = state_preds.mean(dim=0), state_preds.var(dim=0)
        tr_means, tr_vars = latents.mean(dim=0), latents.var(dim=0)
        mean_loss = F.mse_loss(sp_means, tr_means)
        var_loss = F.mse_loss(sp_vars, tr_vars)
        return {self.output: mean_loss, self.var_loss_key: var_loss}


class RandomScaleInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.min_samples = opt['min_samples']

    def forward(self, state):
        inp = state[self.input]
        if self.min_samples < inp.shape[-1]:
            samples = random.randint(self.min_samples, inp.shape[-1])
            start = random.randint(0, inp.shape[-1]-samples)
            inp = inp[:, :, start:start+samples]
        return {self.output: inp}


def pixel_shuffle_1d(x, upscale_factor):
    batch_size, channels, steps = x.size()
    channels //= upscale_factor
    input_view = x.contiguous().view(batch_size, channels, upscale_factor, steps)
    shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
    return shuffle_out.view(batch_size, channels, steps * upscale_factor)


def pixel_unshuffle_1d(x, downscale):
    b, c, s = x.size()
    x = x.view(b, c, s//downscale, downscale)
    x = x.permute(0,1,3,2).contiguous()
    x = x.view(b, c*downscale, s//downscale)
    return x


class AudioUnshuffleInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.compression = opt['compression']

    def forward(self, state):
        inp = state[self.input]
        return {self.output: pixel_unshuffle_1d(inp, self.compression)}

class ClvpTextInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from scripts.audio.gen.speech_synthesis_utils import load_clvp
        self.clvp = load_clvp()
        del self.clvp.speech_transformer  # We will only be using the text transformer.
        self.needs_move = True

    def forward(self, state):
        codes = state[self.input]
        with torch.no_grad():
            if self.needs_move:
                self.clvp = self.clvp.to(codes.device)
            latents = self.clvp.embed_text(codes)
            return {self.output: latents}


class NormalizeMelInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)

    def forward(self, state):
        mel = state[self.input]
        with torch.no_grad():
            return {self.output: normalize_mel(mel)}


class ChannelClipInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.lo = opt['channel_low']
        self.hi = opt['channel_high']

    def forward(self, state):
        inp = state[self.input]
        return {self.output: inp[:,self.lo:self.hi]}


class MusicCheaterLatentInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from models.audio.music.gpt_music2 import UpperEncoder
        self.encoder = UpperEncoder(256, 1024, 256).eval()
        self.encoder.load_state_dict(torch.load('../experiments/music_cheater_encoder_256.pth', map_location=torch.device('cpu')))

    def forward(self, state):
        with torch.no_grad():
            mel = state[self.input]
            self.encoder = self.encoder.to(mel.device)
            proj = self.encoder(mel)
            return {self.output: proj}


class KmeansQuantizerInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        _, self.centroids = torch.load(opt['centroids'])
        k, b = self.centroids.shape
        self.centroids = self.centroids.permute(1,0)

    def forward(self, state):
        with torch.no_grad():
            x = state[self.input]
            self.centroids = self.centroids.to(x.device)
            b, c, s = x.shape
            x = x.permute(0,2,1).reshape(b*s, c)
            distances = x.pow(2).sum(1,keepdim=True) - 2 * x @ self.centroids + self.centroids.pow(2).sum(0, keepdim=True)
            distances[distances.isnan()] = 9999999999
            distances = distances.reshape(b, s, self.centroids.shape[-1])
            labels = distances.argmin(-1)
            return {self.output: labels}
