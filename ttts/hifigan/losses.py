from typing import Dict, Union

import torch
from torch import nn
from torch.nn import functional as F
import librosa


class TorchSTFT(nn.Module):  # pylint: disable=abstract-method
    """Some of the audio processing funtions using Torch for faster batch processing.

    Args:

        n_fft (int):
            FFT window size for STFT.

        hop_length (int):
            number of frames between STFT columns.

        win_length (int, optional):
            STFT window length.

        pad_wav (bool, optional):
            If True pad the audio with (n_fft - hop_length) / 2). Defaults to False.

        window (str, optional):
            The name of a function to create a window tensor that is applied/multiplied to each frame/window. Defaults to "hann_window"

        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        n_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        use_mel (bool, optional):
            If True compute the melspectrograms otherwise. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to False.

        spec_gain (float, optional):
            gain applied when converting amplitude to DB. Defaults to 1.0.

        power (float, optional):
            Exponent for the magnitude spectrogram, e.g., 1 for energy, 2 for power, etc.  Defaults to None.

        use_htk (bool, optional):
            Use HTK formula in mel filter instead of Slaney.

        mel_norm (None, 'slaney', or number, optional):
            If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization).

            If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
            See `librosa.util.normalize` for a full description of supported norm values
            (including `+-np.inf`).

            Otherwise, leave all the triangles aiming for a peak value of 1.0. Defaults to "slaney".
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        pad_wav=False,
        window="hann_window",
        sample_rate=None,
        mel_fmin=0,
        mel_fmax=None,
        n_mels=80,
        use_mel=False,
        do_amp_to_db=False,
        spec_gain=1.0,
        power=None,
        use_htk=False,
        mel_norm="slaney",
        normalized=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_wav = pad_wav
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = n_mels
        self.use_mel = use_mel
        self.do_amp_to_db = do_amp_to_db
        self.spec_gain = spec_gain
        self.power = power
        self.use_htk = use_htk
        self.mel_norm = mel_norm
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.mel_basis = None
        self.normalized = normalized
        if use_mel:
            self._build_mel_basis()

    def __call__(self, x):
        """Compute spectrogram frames by torch based stft.

        Args:
            x (Tensor): input waveform

        Returns:
            Tensor: spectrogram frames.

        Shapes:
            x: [B x T] or [:math:`[B, 1, T]`]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.pad_wav:
            padding = int((self.n_fft - self.hop_length) / 2)
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
        # B x D x T x 2
        o = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(x.device),
            center=True,
            pad_mode="reflect",  # compatible with audio.py
            normalized=self.normalized,
            onesided=True,
            return_complex=False,
        )
        M = o[:, :, :, 0]
        P = o[:, :, :, 1]
        S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))

        if self.power is not None:
            S = S**self.power

        if self.use_mel:
            S = torch.matmul(self.mel_basis.to(x), S)
        if self.do_amp_to_db:
            S = self._amp_to_db(S, spec_gain=self.spec_gain)
        return S

    def _build_mel_basis(self):
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=self.use_htk,
            norm=self.mel_norm,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    @staticmethod
    def _amp_to_db(x, spec_gain=1.0):
        return torch.log(torch.clamp(x, min=1e-5) * spec_gain)

    @staticmethod
    def _db_to_amp(x, spec_gain=1.0):
        return torch.exp(x) / spec_gain
#################################
# GENERATOR LOSSES
#################################


class STFTLoss(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(n_fft, hop_length, win_length)

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, n_ffts=(1024, 2048, 512), hop_lengths=(120, 240, 50), win_lengths=(600, 1200, 240)):
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc


class L1SpecLoss(nn.Module):
    """L1 Loss over Spectrograms as described in HiFiGAN paper https://arxiv.org/pdf/2010.05646.pdf"""

    def __init__(
        self, sample_rate, n_fft, hop_length, win_length, mel_fmin=None, mel_fmax=None, n_mels=None, use_mel=True
    ):
        super().__init__()
        self.use_mel = use_mel
        self.stft = TorchSTFT(
            n_fft,
            hop_length,
            win_length,
            sample_rate=sample_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            n_mels=n_mels,
            use_mel=use_mel,
        )

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        return loss_mag


class MultiScaleSubbandSTFTLoss(MultiScaleSTFTLoss):
    """Multiscale STFT loss for multi band model outputs.
    From MultiBand-MelGAN paper https://arxiv.org/abs/2005.05106"""

    # pylint: disable=no-self-use
    def forward(self, y_hat, y):
        y_hat = y_hat.view(-1, 1, y_hat.shape[2])
        y = y.view(-1, 1, y.shape[2])
        return super().forward(y_hat.squeeze(1), y.squeeze(1))


class MSEGLoss(nn.Module):
    """Mean Squared Generator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_real):
        loss_fake = F.mse_loss(score_real, score_real.new_ones(score_real.shape))
        return loss_fake


class HingeGLoss(nn.Module):
    """Hinge Discriminator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_real):
        # TODO: this might be wrong
        loss_fake = torch.mean(F.relu(1.0 - score_real))
        return loss_fake


##################################
# DISCRIMINATOR LOSSES
##################################


class MSEDLoss(nn.Module):
    """Mean Squared Discriminator Loss"""

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.MSELoss()

    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real):
        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape))
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class HingeDLoss(nn.Module):
    """Hinge Discriminator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real):
        loss_real = torch.mean(F.relu(1.0 - score_real))
        loss_fake = torch.mean(F.relu(1.0 + score_fake))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class MelganFeatureLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.L1Loss()

    # pylint: disable=no-self-use
    def forward(self, fake_feats, real_feats):
        loss_feats = 0
        num_feats = 0
        for idx, _ in enumerate(fake_feats):
            for fake_feat, real_feat in zip(fake_feats[idx], real_feats[idx]):
                loss_feats += self.loss_func(fake_feat, real_feat)
                num_feats += 1
        loss_feats = loss_feats / num_feats
        return loss_feats


#####################################
# LOSS WRAPPERS
#####################################


def _apply_G_adv_loss(scores_fake, loss_func):
    """Compute G adversarial loss function
    and normalize values"""
    adv_loss = 0
    if isinstance(scores_fake, list):
        for score_fake in scores_fake:
            fake_loss = loss_func(score_fake)
            adv_loss += fake_loss
        adv_loss /= len(scores_fake)
    else:
        fake_loss = loss_func(scores_fake)
        adv_loss = fake_loss
    return adv_loss


def _apply_D_loss(scores_fake, scores_real, loss_func):
    """Compute D loss func and normalize loss values"""
    loss = 0
    real_loss = 0
    fake_loss = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake, score_real in zip(scores_fake, scores_real):
            total_loss, real_loss_, fake_loss_ = loss_func(score_fake=score_fake, score_real=score_real)
            loss += total_loss
            real_loss += real_loss_
            fake_loss += fake_loss_
        # normalize loss values with number of scales (discriminators)
        loss /= len(scores_fake)
        real_loss /= len(scores_real)
        fake_loss /= len(scores_fake)
    else:
        # single scale loss
        total_loss, real_loss, fake_loss = loss_func(scores_fake, scores_real)
        loss = total_loss
    return loss, real_loss, fake_loss


##################################
# MODEL LOSSES
##################################


class GeneratorLoss(nn.Module):
    """Generator Loss Wrapper. Based on model configuration it sets a right set of loss functions and computes
    losses. It allows to experiment with different combinations of loss functions with different models by just
    changing configurations.

    Args:
        C (AttrDict): model configuration.
    """

    def __init__(self):
        super().__init__()

        self.use_stft_loss = False
        self.use_subband_stft_loss = False
        self.use_mse_gan_loss = True
        self.use_hinge_gan_loss = False
        self.use_feat_match_loss = True
        self.use_l1_spec_loss = True

        self.stft_loss_weight = 0
        self.subband_stft_loss_weight = 0
        self.mse_gan_loss_weight = 1
        self.hinge_gan_loss_weight = 0
        self.feat_match_loss_weight = 108
        self.l1_spec_loss_weight = 45

        self.mse_loss = MSEGLoss()
        self.feat_match_loss = MelganFeatureLoss()
        self.l1_spec_loss = L1SpecLoss(**{
            "use_mel": True,
            "sample_rate": 24000,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 100,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        })

    def forward(
        self, y_hat=None, y=None, scores_fake=None, feats_fake=None, feats_real=None, y_hat_sub=None, y_sub=None
    ):
        gen_loss = 0
        adv_loss = 0
        return_dict = {}

        # STFT Loss
        if self.use_stft_loss:
            stft_loss_mg, stft_loss_sc = self.stft_loss(y_hat[:, :, : y.size(2)].squeeze(1), y.squeeze(1))
            return_dict["G_stft_loss_mg"] = stft_loss_mg
            return_dict["G_stft_loss_sc"] = stft_loss_sc
            gen_loss = gen_loss + self.stft_loss_weight * (stft_loss_mg + stft_loss_sc)

        # L1 Spec loss
        if self.use_l1_spec_loss:
            l1_spec_loss = self.l1_spec_loss(y_hat, y)
            return_dict["G_l1_spec_loss"] = l1_spec_loss
            gen_loss = gen_loss + self.l1_spec_loss_weight * l1_spec_loss

        # subband STFT Loss
        if self.use_subband_stft_loss:
            subband_stft_loss_mg, subband_stft_loss_sc = self.subband_stft_loss(y_hat_sub, y_sub)
            return_dict["G_subband_stft_loss_mg"] = subband_stft_loss_mg
            return_dict["G_subband_stft_loss_sc"] = subband_stft_loss_sc
            gen_loss = gen_loss + self.subband_stft_loss_weight * (subband_stft_loss_mg + subband_stft_loss_sc)

        # multiscale MSE adversarial loss
        if self.use_mse_gan_loss and scores_fake is not None:
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mse_loss)
            return_dict["G_mse_fake_loss"] = mse_fake_loss
            adv_loss = adv_loss + self.mse_gan_loss_weight * mse_fake_loss

        # multiscale Hinge adversarial loss
        if self.use_hinge_gan_loss and not scores_fake is not None:
            hinge_fake_loss = _apply_G_adv_loss(scores_fake, self.hinge_loss)
            return_dict["G_hinge_fake_loss"] = hinge_fake_loss
            adv_loss = adv_loss + self.hinge_gan_loss_weight * hinge_fake_loss

        # Feature Matching Loss
        if self.use_feat_match_loss and not feats_fake is None:
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            return_dict["G_feat_match_loss"] = feat_match_loss
            adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss
        return_dict["loss"] = gen_loss + adv_loss
        return_dict["G_gen_loss"] = gen_loss
        return_dict["G_adv_loss"] = adv_loss
        return return_dict


class DiscriminatorLoss(nn.Module):
    """Like ```GeneratorLoss```"""

    def __init__(self):
        super().__init__()

        self.use_mse_gan_loss = True
        self.mse_loss = MSEDLoss()

    def forward(self, scores_fake, scores_real):
        loss = 0
        return_dict = {}

        if self.use_mse_gan_loss:
            mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake, scores_real=scores_real, loss_func=self.mse_loss
            )
            return_dict["D_mse_gan_loss"] = mse_D_loss
            return_dict["D_mse_gan_real_loss"] = mse_D_real_loss
            return_dict["D_mse_gan_fake_loss"] = mse_D_fake_loss
            loss += mse_D_loss

        return_dict["loss"] = loss
        return return_dict
