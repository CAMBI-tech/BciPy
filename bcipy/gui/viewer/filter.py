"""Defines filter functions that can be applied to a viewer stream."""
import numpy as np
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


def sig_pro_filter(factor, fs):
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
        return data[:, ::factor]
    return fn
