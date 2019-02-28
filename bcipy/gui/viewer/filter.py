"""Defines filter functions that can be applied to a viewer stream."""
import numpy as np
from bcipy.signal.process.filter import bandpass, notch
# from bcipy.signal.processing.sig_pro import sig_pro


def downsample_filter(factor: int, fs: int):
    """Returns a downsample filter with the given factor.
    Parameters:
    -----------
        factor - downsample factor
        fs - samples per second
    Returns:
    --------
        a function that downsamples the provided data.
    """

    def fn(data):
        """Data should be an np array with a row (np array: float) for each
        channel."""
        # take every nth element of each row
        return data[:, ::factor]
        # return np.array(data)[::factor]
    return fn


def sig_pro_filter(factor,
                   fs,
                   notch_filter_frequency=60,
                   filter_low=2,
                   filter_high=45,
                   filter_order=2):
    """Data filter that calls the sig_pro filter.
    Parameters:
    -----------
        factor - downsample factor
        fs - samples per second
    Returns:
    --------
        a function that filters the provided data using sig_pro.
    """
    # TODO:
    def fn(data):
        """Data should be an np array with a row (np array: float) for each
        channel."""
        notch_filtered_data = notch.notch_filter(data, fs, notch_filter_frequency)
        filtered_data = bandpass.butter_bandpass_filter(
            notch_filtered_data, filter_low, filter_high, fs, order=filter_order)
        # return downsampled, filtered data
        return filtered_data[:, ::factor]
    return fn
