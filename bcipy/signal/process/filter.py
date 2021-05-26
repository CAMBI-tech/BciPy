from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfilt


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
