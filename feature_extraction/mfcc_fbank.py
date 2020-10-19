import torch
import torchvision
import platform
import numpy as np
from torch import (
    Tensor,
    FloatTensor
)

class MFCC(object):
    """
    Create the Mel-frequency cepstrum coefficients (MFCCs) from an audio signal.

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mfcc (int):  Number of mfc coefficients to retain. (Default: 40)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
    """
    def __init__(self, sample_rate=16000, n_mfcc=40, frame_length=20, frame_shift=10, feature_extract_by='librosa'):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = int(round(int(sample_rate) * 0.001 * int(frame_length)))
        self.hop_length = int(round(int(sample_rate) * 0.001 * int(frame_shift)))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'torchaudio':
            # torchaudio is only supported on Linux (Linux, Mac)
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            import torchaudio

            self.transforms = torchaudio.transforms.MFCC(
                sample_rate=sample_rate, n_mfcc=n_mfcc,
                log_mels=True, win_length=frame_length,
                hop_length=self.hop_length, n_fft=self.n_fft
            )
        else:
            import librosa
            self.transforms = librosa.feature.mfcc

    def __call__(self, signal):
        if self.feature_extract_by == 'torchaudio':
            mfcc = self.transforms(FloatTensor(signal))
            mfcc = mfcc.numpy()

        elif self.feature_extract_by == 'librosa':
            mfcc = self.transforms(
                y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )

        else:
            raise ValueError("Unsupported library : {0}".format(self.feature_extract_by))

        return mfcc


class FilterBank(object):
    """
    Create a fbank from a raw audio signal. This matches the input/output of Kaldi’s compute-fbank-feats

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 80)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
    """
    def __init__(self, sample_rate=16000, n_mels=80, frame_length=20, frame_shift=10):
        import torchaudio
        self.transforms = torchaudio.compliance.kaldi.fbank
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return self.transforms(
            Tensor(signal).unsqueeze(0), num_mel_bins=self.n_mels,
            frame_length=self.frame_length, frame_shift=self.frame_shift,
            window_type='hamming'
        ).transpose(0, 1).numpy()