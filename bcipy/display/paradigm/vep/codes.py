"""Functions related to VEP flash rates"""
import logging
from typing import List

import numpy as np

from bcipy.helpers.exceptions import BciPyCoreException

log = logging.getLogger(__name__)


def create_vep_codes(length=32, count=4) -> List[List[int]]:
    """Create a list of random VEP codes.

    length - how many bits in each code. This should be greater than or equal to the refresh rate
        if using these to flicker. For example, if the refresh rate is 60Hz, then the length should
        be at least 60.
    count - how many codes to generate, each will be unique.
    """
    np.random.seed(1)
    return [np.random.randint(2, size=length) for _ in range(count)]


def ssvep_to_code(refresh_rate: int = 60, flicker_rate: int = 10) -> List[int]:
    """Convert SSVEP to Code.

    Converts a SSVEP (steady state visual evoked potential; ex. 10 Hz) to a code (0,1)
    given the refresh rate of the monitor (Hz) provided and a desired flicker rate (Hz).

    Parameters:
    -----------
        refresh_rate: int, refresh rate of the monitor (Hz)
        flicker_rate: int, desired flicker rate (Hz)
    Returns:
    --------
        list of 0s and 1s that represent the code for the SSVEP on the monitor.
    """
    if flicker_rate > refresh_rate:
        raise BciPyCoreException(
            'flicker rate cannot be greater than refresh rate')
    if flicker_rate <= 1:
        raise BciPyCoreException('flicker rate must be greater than 1')

    # get the number of frames per flicker
    length_flicker = refresh_rate / flicker_rate

    if length_flicker.is_integer():
        length_flicker = int(length_flicker)
    else:
        err_message = f'flicker rate={flicker_rate} is not an integer multiple of refresh rate={refresh_rate}'
        log.exception(err_message)
        raise BciPyCoreException(err_message)

    # start the first frames as off (0) for length of flicker;
    # it will then toggle on (1)/ off (0) for length of flicker until all frames are filled for refresh rate.
    t = 0
    codes = []
    for _ in range(flicker_rate):
        codes += [t] * length_flicker
        t = 1 - t

    return codes
