from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .peq import ParametricEqualizer
from .praat import PraatAugment


class Augment(nn.Module):
    """Waveform augmentation.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Nansy configurations.
        """
        super().__init__()
        self.config = config
        self.praat = PraatAugment(config)
        self.peq = ParametricEqualizer(
            config.data.sampling_rate, config.data.win_length)
        self.register_buffer(
            'window',
            torch.hann_window(config.data.win_length),
            persistent=False)
        f_min, f_max, peaks = \
            config.train.cutoff_lowpass, \
            config.train.cutoff_highpass, config.train.num_peak
        # peaks except frequency min and max
        self.register_buffer(
            'peak_centers',
            f_min * (f_max / f_min) ** (torch.arange(peaks + 2)[1:-1] / (peaks + 1)),
            persistent=False)

    def forward(self,
                wavs: torch.Tensor,
                pitch_shift: Optional[torch.Tensor] = None,
                pitch_range: Optional[torch.Tensor] = None,
                formant_shift: Optional[torch.Tensor] = None,
                quality_power: Optional[torch.Tensor] = None,
                gain: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Augment the audio signal, random pitch, formant shift and PEQ.
        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            pitch_shift: [torch.float32; [B]], pitch shifts.
            pitch_range: [torch.float32; [B]], pitch ranges.
            formant_shift: [torch.float32; [B]], formant shifts.
            quality_power: [torch.float32; [B, num_peak + 2]],
                exponents of quality factor, for PEQ.
            gain: [torch.float32; [B, num_peak + 2]], gain in decibel.
        Returns:
            [torch.float32; [B, T]], augmented.
        """
        # B
        bsize, _ = wavs.shape
        # [B, F, T / S], complex64
        fft = torch.stft(
            wavs,
            self.config.data.win_length,
            self.config.data.hop_length,
            self.config.data.win_length,
            self.window,
            return_complex=True)
        # PEQ
        if quality_power is not None:
            # alias
            q_min, q_max = self.config.train.q_min, self.config.train.q_max
            # [B, num_peak + 2]
            q = q_min * (q_max / q_min) ** quality_power
            if gain is None:
                # [B, num_peak]
                gain = torch.zeros_like(q[:, :-2])
            # [B, num_peak]
            center = self.peak_centers[None].repeat(bsize, 1)
            # [B, F]
            peaks = torch.prod(
                self.peq.peaking_equalizer(center, gain[:, :-2], q[:, :-2]), dim=1)
            # [B, F]
            lowpass = self.peq.low_shelving(
                self.config.train.cutoff_lowpass, gain[:, -2], q[:, -2])
            highpass = self.peq.high_shelving(
                self.config.train.cutoff_highpass, gain[:, -1], q[:, -1])
            # [B, F]
            filters = peaks * highpass * lowpass
            # [B, F, T / S]
            fft = fft * filters[..., None]
        # [B, T]
        out = torch.istft(
            fft,
            self.config.data.win_length,
            self.config.data.hop_length,
            self.config.data.win_length,
            self.window).clamp(-1., 1.)
        # max value normalization
        out = out / out.abs().amax(dim=-1, keepdim=True).clamp_min(1e-7)
        if formant_shift is None and pitch_shift is None and pitch_range is None:
            return out
        # praat-based augmentation
        if formant_shift is None:
            formant_shift = torch.ones(bsize)
        if pitch_shift is None:
            pitch_shift = torch.ones(bsize)
        if pitch_range is None:
            pitch_range = torch.ones(bsize)
        out = torch.tensor(
            np.stack([
                self.praat.augment(o, fs.item(), ps.item(), pr.item())
                for o, fs, ps, pr in zip(
                    out.cpu().numpy(),
                    formant_shift.cpu().numpy(),
                    pitch_shift.cpu().numpy(),
                    pitch_range.cpu().numpy())], axis=0),
            device=out.device, dtype=torch.float32)
        return out
