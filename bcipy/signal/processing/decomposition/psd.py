import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from enum import Enum
from scipy import signal
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper


class PSD_TYPE(Enum):
    WELCH = 'Welch'
    MULTITAPER = 'MutliTaper'


def power_spectral_density(
        data: np.ndarray,
        band: tuple,
        fs: float=100,
        window_length: float=4,
        plot: bool=True,
        method: PSD_TYPE=PSD_TYPE.MULTITAPER,
        relative=False):

    """Power spectral density:

    Many thanks to: https://raphaelvallat.github.io/bandpower.html
    """

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == PSD_TYPE.WELCH:
        if window_length is not None:
            nperseg = window_length * fs
        else:
            nperseg = (2 / low) * fs

        freqs, psd = welch(data, fs, nperseg=nperseg, scaling='density')

    elif method == PSD_TYPE.MULTITAPER:
        psd, freqs = psd_array_multitaper(data, fs, adaptive=True, normalization='full')

    # Find index of band in frequency vector
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx_band = np.zeros(dtype=bool, shape=freqs.shape)
    idx_band[idx_min:idx_max] = True

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], freqs[idx_band])

    # Define sampling frequency and time vector
    np.arange(data.size) / fs
    win = window_length * fs

    # Plot the power spectrum
    if plot:
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, color='k', lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title(f'{method.value}')
        plt.xlim([0, 20])
        sns.despine()
        plt.show()

    if relative:
        bp /= simps(psd, freqs)
    return bp


if __name__ == '__main__':
    data = np.loadtxt('bcipy/signal/processing/decomposition/data.txt')
    fs = 100
    band = (0, 100)
    np.arange(data.size) / fs
    power_spectral_density(data, band, fs=fs, method=PSD_TYPE.WELCH)
