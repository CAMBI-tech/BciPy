"""Functions related to VEP flash rates"""
import logging
from typing import List

import numpy as np
from scipy.signal import max_len_seq
from bcipy.helpers.exceptions import BciPyCoreException

log = logging.getLogger(__name__)

# These rates work for a 60hz display
DEFAULT_FLICKER_RATES = [4, 5, 6, 10, 12, 15, 20, 30]


def mseq(seed, taps):
    """
    Generate an LFSR sequence.

    :param seed: Initial state of the LFSR as a list of bits (0 or 1).
    :param taps: Positions in the LFSR to be XORed, given as a list of indices.
    :return: List representing the generated LFSR sequence.
    """
    state = seed.copy()
    # sequence = []

    # for _ in range(2 ** (len(seed)) - 1):
    #     # Output the first bit of the state
    #     sequence.append(state[0])

    #     # Calculate the feedback bit
    #     feedback = 0
    #     for tap in taps:
    #         feedback ^= state[tap]

    #     # Shift the register and insert the feedback bit at the end
    #     state = state[1:] + [feedback]

    # return sequence

    sequence, _ = max_len_seq(nbits = len(state), taps = taps, state = state)

    return sequence.tolist()


def create_vep_codes(length=63, count=8, seed=[1, 0, 0, 1, 1, 0, 1], taps=[1, 3, 5], shift_by: int = 5) -> List[List[int]]:
    """Create a list of VEP codes using m-sequence (LFSR sequence).

    length - how many bits in each code. This should be greater than or equal to the refresh rate
        if using these to flicker. For example, if the refresh rate is 60Hz, then the length should
        be at least 60.
    count - how many codes to generate, each will be unique.
    seed - Initial state of the LFSR.
    taps - Tap positions in the LFSR to generate feedback.
    """
    #generates the original
    original_mseq = mseq(seed, taps)

    codes = []
    
    #create unique codes by applying cyclical shifts
    for i in range(count):
        shift = (i * shift_by) % len(original_mseq)  #determine the shift based on index
        shifted_sequence = original_mseq[shift:] + original_mseq[:shift]  #apply the cyclical shift
        codes.append(shifted_sequence[:length])  #truncate to the required length if necessary
    
    print(len(codes))
    return codes


def ssvep_to_code(refresh_rate: int = 60, flicker_rates: List[int] = DEFAULT_FLICKER_RATES) -> List[List[int]]:
    """Convert SSVEP to Code.

    Converts a SSVEP (steady state visual evoked potential; ex. 10 Hz) to a code (0,1)
    given the refresh rate of the monitor (Hz) provided and a desired flicker rate (Hz).

    TODO: https://www.pivotaltracker.com/story/show/186522657
    Consider an additional parameter for the number of seconds.

    Parameters:
    -----------
        refresh_rate: int, refresh rate of the monitor (Hz)
        flicker_rate: int, desired flicker rate (Hz)
    Returns:
    --------
        list of 0s and 1s that represent the code for the SSVEP on the monitor.
    """
    # if flicker_rate > refresh_rate:
    #     raise BciPyCoreException(
    #         'flicker rate cannot be greater than refresh rate')
    # if flicker_rate <= 1:
    #     raise BciPyCoreException('flicker rate must be greater than 1')

    # # get the number of frames per flicker
    # length_flicker = refresh_rate / flicker_rate

    # if length_flicker.is_integer():
    #     length_flicker = int(length_flicker)
    # else:
    #     err_message = f'flicker rate={flicker_rate} is not an integer multiple of refresh rate={refresh_rate}'
    #     log.exception(err_message)
    #     raise BciPyCoreException(err_message)

    # # start the first frames as off (0) for length of flicker;
    # # it will then toggle on (1)/ off (0) for length of flicker until all frames are filled for refresh rate.
    # t = 0
    # codes = []
    # for _ in range(flicker_rate):
    #     codes += [t] * length_flicker
    #     t = 1 - t

    # return codes

    codes = []
    for flicker_rate in flicker_rates:
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
        code = []
        for _ in range(flicker_rate):
            code += [t] * length_flicker
            t = 1 - t
        codes.append(code)

    return codes


def round_refresh_rate(rate: float) -> int:
    """Round the given display rate to the nearest 10s value.

    >>> round_refresh_rate(59.12)
    60
    >>> round_refresh_rate(61.538)
    60
    >>> round_refresh_rate(121.23)
    120
    """
    return int(round(rate, -1))
