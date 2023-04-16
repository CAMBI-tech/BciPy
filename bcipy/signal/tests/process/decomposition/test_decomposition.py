import unittest

from bcipy.config import BCIPY_ROOT
from bcipy.helpers.exceptions import SignalException
from bcipy.signal.process.decomposition import continuous_wavelet_transform
from bcipy.signal.process.decomposition.psd import power_spectral_density, PSD_TYPE
import numpy as np


class TestDecomposition(unittest.TestCase):

    def setUp(self) -> None:
        self.fs = 100
        data = np.loadtxt(f'{BCIPY_ROOT}/signal/tests/process/decomposition/resources/data.txt')
        np.arange(data.size) / self.fs
        self.data = data
        self.band = (8, 12)

    def test_cwt(self):
        # create test data to pass to cwt
        # test data should be 3d array of shape (trials, channels, time)
        data = np.random.rand(10, 2, 1000)
        freq = 8
        fs = 100
        wavelet = "cmor1.5-1.0"
        data, scales = continuous_wavelet_transform(data, freq, fs, wavelet)
        self.assertEqual(data.shape, (10, 2, 1000))
        self.assertEqual(scales, fs / freq)

    def test_cwt_with_bad_wavelet(self):
        data = np.random.rand(10, 2, 1000)
        freq = 8
        fs = 100
        wavelet = "bad_wavelet"
        with self.assertRaises(ValueError):
            continuous_wavelet_transform(data, freq, fs, wavelet)

    def test_psd_relative(self):
        response = power_spectral_density(
            self.data,
            self.band,
            sampling_rate=self.fs,
            relative=True,
            plot=False)

        self.assertIsInstance(response, float)
        self.assertTrue(0 <= response <= 1)

    def test_psd_welch(self):
        response = power_spectral_density(
            self.data,
            self.band,
            sampling_rate=self.fs,
            method=PSD_TYPE.WELCH,
            plot=False)

        self.assertIsInstance(response, float)

    def test_psd_multitaper(self):
        response = power_spectral_density(
            self.data,
            self.band,
            sampling_rate=self.fs,
            method=PSD_TYPE.MULTITAPER,
            plot=False)

        self.assertIsInstance(response, float)

    def test_psd_bad_method(self):
        with self.assertRaises(SignalException):
            power_spectral_density(
                self.data,
                self.band,
                sampling_rate=self.fs,
                method="bad_method",
                plot=False)

    def test_psd_bad_band(self):
        # create test data to pass to psd low band > high band
        band = (12, 8)

        with self.assertRaises(IndexError):
            power_spectral_density(
                self.data,
                band,
                sampling_rate=self.fs,
                plot=False)

    def test_psd_bad_data(self):
        data = np.random.rand(10, 2, 1000)  # no frequency dimension

        with self.assertRaises(IndexError):
            power_spectral_density(
                data,
                self.band,
                sampling_rate=self.fs,
                plot=False)


if __name__ == '__main__':
    unittest.main()
