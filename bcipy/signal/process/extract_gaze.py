import numpy as np

from bcipy.helpers.exceptions import SignalException


def extract_eye_info(data):
    """"Rearrange the dimensions of gaze inquiry data and reshape it to num_channels x num_samples
    Extract Left and Right Eye info from data. Remove all blinks, do necessary preprocessing.
    The data is extracted according to the channel map:
    ['device_ts, 'system_ts', 'left_x', 'left_y', 'left_pupil', 'right_x', 'right_y', 'right_pupil']

    Args:
        data (np.ndarray): Data in shape of num_channels x num_samples

    Returns:
        left_eye (np.ndarray), left_pupil (List(float))
        right_eye (np.ndarray), right_pupil (List(float))
    """

    # Extract samples from channels
    lx = data[2, :]
    ly = data[3, :]
    left_pupil = data[4, :]  # Use if needed

    rx = data[5, :]
    ry = data[6, :]
    right_pupil = data[7, :]  # Use if needed

    left_eye = np.vstack((np.array(lx), np.array(ly))).T
    right_eye = np.vstack((np.array(rx), np.array(ry))).T

    # Remove ALL blinks (i.e. Nan values) regardless of which eye it occurs.
    # Make sure that the number of samples are the same for both eyes
    left_eye_nan_idx = np.isnan(left_eye).any(axis=1)
    deleted_samples = left_eye_nan_idx.sum()
    all_samples = len(left_eye)
    left_eye = left_eye[~left_eye_nan_idx]
    right_eye = right_eye[~left_eye_nan_idx]
    left_pupil = left_pupil[~left_eye_nan_idx]
    right_pupil = right_pupil[~left_eye_nan_idx]

    right_eye_nan_idx = np.isnan(right_eye).any(axis=1)
    left_eye = left_eye[~right_eye_nan_idx]
    right_eye = right_eye[~right_eye_nan_idx]
    left_pupil = left_pupil[~right_eye_nan_idx]
    right_pupil = right_pupil[~right_eye_nan_idx]

    try:
        len(left_eye) != len(right_eye)
    except AssertionError:
        raise SignalException(
            'Number of samples for left and right eye are not the same.')

    return left_eye, right_eye, left_pupil, right_pupil, deleted_samples, all_samples
