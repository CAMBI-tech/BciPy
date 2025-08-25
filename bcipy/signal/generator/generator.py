from typing import List

import numpy as np


def truncate_float(num: float, precision: int) -> float:
    """Truncate a float to a given precision."""
    return float(str(num)[:precision])


def gen_random_data(
        low: float,
        high: float,
        channel_count: int,
        precision: int = 8) -> List[float]:
    """Generate random data.

    This function generates random data for testing purposes within a given range. The data is
    generated with a given precision. The default precision is 8. The data is generated using the
    numpy.random.uniform function. The data is truncated to the given precision using the truncate_float
    function. In order to generate a full session of data, this function can be called multiple times.

    Args:
        low (float): Lower bound of the random data.
        high (float): Upper bound of the random data.
        channel_count (int): Number of channels to generate.
        precision (int): Precision of the random data.

    Returns:
        list: List of random data.
    """
    return [truncate_float(np.random.uniform(low, high), precision)
            for _ in range(channel_count)]
