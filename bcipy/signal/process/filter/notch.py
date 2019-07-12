import numpy as np
from scipy.signal import filtfilt, iirnotch


def notch_filter(data: np.ndarray, fs: int, frequency_to_remove: int,
                 quality_factor: int = 30) -> np.ndarray:
    """Notch Filter.

    A notch filter is a bandstop filter with a narrow bandwidth. It removes
        the frequency of interest without much impact to others.
    """
    normalized_frequency = frequency_to_remove / (fs / 2)
    b, a = iirnotch(normalized_frequency, quality_factor)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
