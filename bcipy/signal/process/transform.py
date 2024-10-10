from typing import NamedTuple, Tuple

import numpy as np

from bcipy.signal.process.filter import Bandpass, Notch


class Composition:
    """Applies a sequence of transformations"""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        for transform in self.transforms:
            data, fs = transform(data, fs)
        return data, fs


class Downsample:
    """Downsampling by an integer factor"""

    def __init__(self, factor: int = 2, *args, **kwargs):
        self.factor = factor

    def __call__(self, data: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        return data[:, :: self.factor], fs // self.factor


class ERPTransformParams(NamedTuple):
    """Parameters used for the default transform."""
    notch_filter_frequency: int = 60
    filter_low: int = 2
    filter_high: int = 45
    filter_order: int = 2
    down_sampling_rate: int = 2

    def __str__(self):
        return ' '.join([
            f"Filter: [{self.filter_low}-{self.filter_high}] Hz \n"
            f"Order: {self.filter_order} \n",
            f"Notch: {self.notch_filter_frequency} Hz \n",
            f"Downsample: {self.down_sampling_rate} \n"
        ])


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
