import numpy as np


def downsample(data: np.array, factor: int = 2):
    """Downsamples the data to the given factor

    Parameters:
    -----------
        data : np.array
        factor: amount to downsample
    Returns:
        downsampled data
    """
    return data[:, ::factor]
