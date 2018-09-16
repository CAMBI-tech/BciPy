import numpy as np

def gen_random_data(low, high, channel_count):
	return [np.random.uniform(low, high)
                   for i in range(channel_count)]