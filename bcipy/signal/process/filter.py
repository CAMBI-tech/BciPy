from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


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
        return sosfiltfilt(self.sos, data), fs


def filter_inquiries(inquiries, transform, sample_rate) -> Tuple[np.ndarray, float]:
    """Filter Inquiries.

    The shape of data after reshaping into inquiries requires a bit of pre-processing to apply the
        transforms and filters defined in BciPy without looping. Here we flatten the inquires, filter,
        and return them in the correct shape for continued processing.
    """
    # (Channels, Inquiries, Samples)
    old_shape = inquiries.shape
    # (Channels*Inquiry, Samples)
    inq_flatten = inquiries.reshape(-1, old_shape[-1])
    inq_flatten_filtered, transformed_sample_rate = transform(inq_flatten, sample_rate)
    # (Channels, Inquiries, Samples)
    inquiries = inq_flatten_filtered.reshape(*old_shape[:2], inq_flatten_filtered.shape[-1])
    return inquiries, transformed_sample_rate
