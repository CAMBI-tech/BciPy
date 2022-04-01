import numpy as np


def truncate_float(num, precision):
    return float(str(num)[:precision])


def gen_random_data(low, high, channel_count, precision=8):
    return [truncate_float(np.random.uniform(low, high), precision)
            for _ in range(channel_count)]
