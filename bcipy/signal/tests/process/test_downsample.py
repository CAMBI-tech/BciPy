from bcipy.signal.process import Downsample
import numpy as np
import unittest


class TestDownsample(unittest.TestCase):
    """Test downsample functionality"""

    def test_downsample(self):
        data = np.array(np.ones((100, 100)))
        fs_before = 256
        factor = 2
        downsampled_data, fs_after = Downsample(factor=factor)(data, fs_before)
        self.assertEqual(len(downsampled_data[0]), 50)
        self.assertEqual(fs_before // factor, fs_after)
        self.assertEqual(len(downsampled_data[:][0]), 50)
