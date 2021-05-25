from scipy.signal import butter, filtfilt, iirnotch, sosfilt
import numpy as np
from typing import Optional, Tuple


class Composition:
    """Applies a sequence of transformations"""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        for transform in self.transforms:
            data, fs = transform(data, fs)
        return data, fs


class Notch:
    """Remove a single frequency"""

    def __init__(self, sample_rate_hz, remove_freq_hz=60.0, quality_factor=30):
        remove_freq_hz = remove_freq_hz / (sample_rate_hz / 2)
        self.b, self.a = iirnotch(remove_freq_hz, quality_factor)

    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return filtfilt(self.b, self.a, data), fs


class Bandpass:
    """Preserve a specified range of frequencies"""

    def __init__(self, lo, hi, sample_rate_hz, order=5):
        nyq = 0.5 * sample_rate_hz
        lo, hi = lo / nyq, hi / nyq
        self.sos = butter(order, [lo, hi], analog=False, btype="band", output="sos")

    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return sosfilt(self.sos, data), fs


class Downsample:
    """Downsampling by an integer factor"""

    def __init__(self, factor: int = 2):
        self.factor = factor

    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return data[:, :: self.factor], fs // self.factor


def get_default_transform(
    sample_rate_hz: int,
    notch_freq_hz: int,
    bandpass_low: int,
    bandpass_high: int,
    bandpass_order: int,
    downsample_factor: int,
    notch_quality_factor: int = 30,
):
    return Composition(
        Notch(sample_rate_hz, notch_freq_hz, notch_quality_factor),
        Bandpass(bandpass_low, bandpass_high, sample_rate_hz, bandpass_order),
        Downsample(downsample_factor),
    )
