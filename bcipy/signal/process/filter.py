from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, sosfilt
from mne.filter import create_filter, filter_data


class Notch:
    """Remove a single frequency"""

    def __init__(self, sample_rate_hz, remove_freq_hz=60.0, quality_factor=30):
        nyq = 0.5 * sample_rate_hz
        remove_freq_hz = remove_freq_hz / nyq
        self.b, self.a = iirnotch(remove_freq_hz, quality_factor)

    def __call__(self, data: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        return filtfilt(self.b, self.a, data), fs


class Bandpass:
    """Preserve a specified range of frequencies using a Bandpass filter"""

    def __init__(self, lo, hi, sample_rate_hz, order=5):
        nyq = 0.5 * sample_rate_hz
        lo, hi = lo / nyq, hi / nyq
        self.sos = butter(order, [lo, hi], analog=False, btype="band", output="sos")

    def __call__(self, data: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        return sosfiltfilt(self.sos, data), fs
    

class FiniteImpulseResponse:
    """Preserve a specified range of frequencies using an FIR filter"""

    def __init__(self, lo, hi, sample_rate_hz, fir_design='firwin', fir_window='hamming', phase='zero-double'):

        # nyq = 0.5 * sample_rate_hz
        # lo, hi = lo / nyq, hi / nyq
        # self.h = create_filter(
        #     data,
        #     sample_rate_hz,
        #     lo,
        #     hi ,
        #     fir_design=fir_design,
        #     method='fir',
        #     fir_window=fir_window,
        #     phase='zero-double')
        self.window = fir_window
        self.phase = phase
        self.design = fir_design
        self.lo = lo
        self.hi = hi
        self.fs = sample_rate_hz
        
    def __call__(self, data: np.ndarray, fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return self.filter_mne(data), fs
    
    def filter_mne(self, data):
        return filter_data(
            data,
            sfreq=self.fs,
            l_freq=self.lo,
            h_freq=self.hi,
            fir_design=self.design,
            fir_window=self.window,
            phase=self.phase)


def filter_inquiries(inquiries: np.ndarray, transform, sample_rate: int) -> Tuple[np.ndarray, int]:
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
