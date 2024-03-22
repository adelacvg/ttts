from ttts.vqvae.augment import Augment
import torchaudio.functional as AuF
import torchaudio
import json
from ttts.utils.data_utils import HParams
from typing import List, Optional, Tuple, Union
import torch
def sample_like(signal: torch.Tensor) -> List[torch.Tensor]:
    """Sample augmentation parameters.
    Args:
        signal: [torch.float32; [B, T]], speech signal.
    Returns:
        augmentation parameters.
    """
    # [B]
    bsize, _ = signal.shape
    def sampler(ratio):
        shifts = torch.rand(bsize, device=signal.device) * (ratio - 1.) + 1.
        # flip
        flip = torch.rand(bsize) < 0.5
        shifts[flip] = shifts[flip] ** -1
        return shifts
    # sample shifts
    fs = sampler(hps.train.formant_shift)
    ps = sampler(hps.train.pitch_shift)
    pr = sampler(hps.train.pitch_range)
    # parametric equalizer
    peaks = hps.train.num_peak
    # quality factor
    power = torch.rand(bsize, peaks + 2, device=signal.device)
    # gains
    g_min, g_max = hps.train.g_min, hps.train.g_max
    gain = torch.rand(bsize, peaks + 2, device=signal.device) * (g_max - g_min) + g_min
    return fs, ps, pr, power, gain
def augment(signal, aug, ps: bool = False) -> torch.Tensor:
        """Augment the speech.
        Args:
            signal: [torch.float32; [B, T]], segmented speech.
            ps: whether use pitch shift.
        Returns:
            [torch.float32; [B, T]], speech signal.
        """
        # B
        bsize, _ = signal.shape
        saves = None
        while saves is None or len(saves) < bsize:
            # [B] x 4
            fshift, pshift, prange, power, gain = sample_like(signal)
            if not ps:
                pshift = None
            # [B, T]
            out = aug.forward(signal, pshift, prange, fshift, power, gain)
            # for covering unexpected NaN
            nan = out.isnan().any(dim=-1)
            if not nan.all():
                # save the outputs for not-nan inputs
                if saves is None:
                    saves = out[~nan]
                else:
                    saves = torch.cat([saves, out[~nan]], dim=0)
        # [B, T]
        return saves[:bsize]
hps_path='vqvae/config.json'
hps = HParams(**json.load(open(hps_path)))
wav, sr = torchaudio.load('2.wav')
wav32k = AuF.resample(wav, sr, 32000)
aug = Augment(hps)
wav_aug = augment(wav32k, aug)
torchaudio.save('2_aug.wav', wav_aug,32000)