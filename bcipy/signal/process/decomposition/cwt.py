from typing import Tuple

import numpy as np
import pywt


def continuous_wavelet_transform(
        data: np.ndarray, freq: int, fs: int, wavelet="cmor1.5-1.0") -> Tuple[np.ndarray, float]:
    """
    Transform data into frequency domain using Continuous Wavelet Transform (CWT).
    Keeps only a single wavelet scale, specified by `freq`.

    Args:
        data (np.ndarray): shape (trials, channels, time)
        freq (int): frequency of wavelet to keep
        fs (int): sampling rate of data (Hz)
        wavelet (str): name of wavelet to use (see pywt.families() and pywt.wavelist())
            default: "cmor1.5-1.0" (complex morlet wavelet)
            https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#complex-morlet-wavelets

    Returns:
        np.ndarray: frequency domain data, shape (trials, wavelet_scales*channels, time)
                    Note that here we only use 1 wavelet_scale.
        float: wavelet scale used
    """
    scales = pywt.central_frequency(wavelet) * fs / np.array(freq)
    all_coeffs = []
    for trial in data:
        # shape == (scales, channels, time)
        coeffs, _ = pywt.cwt(trial, scales, wavelet)
        all_coeffs.append(coeffs)

    final_data = np.stack(all_coeffs)
    if np.any(np.iscomplex(final_data)):
        final_data = np.abs(final_data) ** 2

    # have shape == (trials, freqs, channels, time)
    # want shape == (trials, freqs*channels, time)
    return final_data.reshape(final_data.shape[0], -1, final_data.shape[-1]), scales
