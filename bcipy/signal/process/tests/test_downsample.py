#from bcipy.signal.process.filter import downsample
from bcipy.signal.process.filter.downsample import downsample
import numpy as np
import unittest

class TestDownsample(unittest.TestCase):
	'''Test downsample functionality'''

	def test_downsample(self):

		data = np.array(np.ones((100,100)))
		downsampled_data = downsample(data)
		self.assertEqual(len(downsampled_data[0]),50)
		self.assertEqual(len(downsampled_data[:][0]),50)