from typing import Optional, Tuple

import numpy as np
from bcipy.signal.process.filter import Notch, Bandpass


class Composition:
    """Applies a sequence of transformations"""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        for transform in self.transforms:
            data, fs = transform(data, fs)
        return data, fs


class Downsample:
    """Downsampling by an integer factor"""

    def __init__(self, factor: int = 2):
        self.factor = factor

    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if fs:
            return data[:, :: self.factor], fs // self.factor
        else:
            return data[:, :: self.factor], None


def get_default_transform(
    sample_rate_hz: int,
    notch_freq_hz: int,
    bandpass_low: int,
    bandpass_high: int,
    bandpass_order: int,
    downsample_factor: int,
    notch_quality_factor: int = 30,
) -> Composition:
    return Composition(
        Notch(sample_rate_hz, notch_freq_hz, notch_quality_factor),
        Bandpass(bandpass_low, bandpass_high, sample_rate_hz, bandpass_order),
        Downsample(downsample_factor),
    )
