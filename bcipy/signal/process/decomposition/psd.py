import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from enum import Enum
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

from typing import Tuple


class PSD_TYPE(Enum):
    """Power Spectal Density Type.

    Enum used to specify the type of power spectral approximation
        to use. They each have trade-offs in terms of computational
        cost and resolution.
    """
    WELCH = 'Welch'
    MULTITAPER = 'MutliTaper'


def power_spectral_density(
        data: np.ndarray,
        band: Tuple[float, float],
        sampling_rate: float = 100.0,
        window_length: float = 4.0,
        plot: bool = False,
        method: PSD_TYPE = PSD_TYPE.WELCH,
        relative=False):
    """Power spectral density:

    Many thanks to: https://raphaelvallat.github.io/bandpower.html

    Parameters
    ----------
        data: Numpy Array.
            Time series data in the form of a numpy ndarray used to estimate the
                power spectral density.
        band: tuple.
            frequency band psd to export. Note this must be within the Nyquist frequency
            for your data. Approximated as sampling rate / 2.
        sampling_rate: float
            Sampling rate of the data in # of samples per second.
        window_length: float
            Length in seconds of data window.
        plot: boolean.
            Whether of not to plot the PSD estimated. Helpful for debugging and exploration.
        method: PSD_TYPE
            Type of PSD estimation method to use during calculation.
        relative: boolean
            Whether or not to express the power in a frequency band as a percentage of the
            total power of the signal.

    """
    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == PSD_TYPE.WELCH:
        nperseg = window_length * sampling_rate
        freqs, psd = welch(data, sampling_rate, nperseg=nperseg)

    # Compute the modified periodogram (MultiTaper)
    elif method == PSD_TYPE.MULTITAPER:
        psd, freqs = psd_array_multitaper(
            data, sampling_rate, adaptive=True, normalization='full', verbose=False)

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    # Plot the power spectrum
    if plot:
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, color='k', lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title(f'{method.value}')
        plt.xlim([0, sampling_rate / 2])
        sns.despine()
        plt.show()

    # Whether or not to return PSD as a percentage of total power
    if relative:
        bp /= simps(psd, dx=freq_res)

    return bp


if __name__ == '__main__':
    data = np.loadtxt('bcipy/signal/process/decomposition/resources/data.txt')
    sampling_rate = 100
    band = (0, 100)
    np.arange(data.size) / sampling_rate
    power_spectral_density(
        data,
        band,
        sampling_rate=sampling_rate,
        method=PSD_TYPE.WELCH,
        plot=True)
