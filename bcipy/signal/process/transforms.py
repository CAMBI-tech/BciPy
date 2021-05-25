from scipy.signal import butter, filtfilt, iirnotch, sosfilt
import numpy as np
from typing import Optional


class Composition:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data: np.ndarray, fs: Optional[int] = None):
        for t in self.transforms:
            data, fs = t(data, fs)
        if fs is None:
            return data
        return data, fs


class Notch:
    def __init__(self, sample_rate_hz, remove_freq_hz=60.0, quality_factor=30):
        remove_freq_hz = remove_freq_hz / (sample_rate_hz / 2)
        self.b, self.a = iirnotch(remove_freq_hz, quality_factor)

    def __call__(self, data: np.ndarray, fs: Optional[int] = None):
        if fs is None:
            return filtfilt(self.b, self.a, data)
        return filtfilt(self.b, self.a, data), fs


class Bandpass:
    def __init__(self, lo, hi, sample_rate_hz, order=5):
        nyq = 0.5 * sample_rate_hz
        lo, hi = lo / nyq, hi / nyq
        self.sos = butter(order, [lo, hi], analog=False, btype="band", output="sos")

    def __call__(self, data: np.ndarray, fs: Optional[int] = None):
        if fs is None:
            return sosfilt(self.sos, data)
        return sosfilt(self.sos, data), fs


class Downsample:
    def __init__(self, factor: int = 2):
        self.factor = factor

    def __call__(self, data: np.ndarray, fs: Optional[int] = None):
        if fs is None:
            return data[:, :: self.factor]
        return data[:, :: self.factor], fs // self.factor


class Lambda:
    def __init__(self, f):
        self.f = f

    def __call__(self, data: np.ndarray, fs: Optional[int] = None):
        if fs is None:
            return self.f(data)
        return self.f(data), fs


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
