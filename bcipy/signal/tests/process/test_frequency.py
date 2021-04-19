import numpy as np
import unittest
from bcipy.signal.process.filter import bandpass, notch


class TestFrequencyFilters(unittest.TestCase):
    def setUp(self):
        frequencies = [2, 10, 20, 30, 40, 45, 50, 60, 70, 80]
        self.sample_rate_hz = 256
        duration_s = 5

        # time indices
        t = np.arange(0, duration_s - 1.0 / self.sample_rate_hz, 1.0 / self.sample_rate_hz)
        self.data = np.zeros_like(t)
        for f in frequencies:
            self.data += np.cos((2 * np.pi * f) * t)  # set frequency to (2*pi*f) and step along each value of t

    def test_notch(self):
        notch_freq_hz = 60
        filtered_data = notch.notch_filter(self.data, self.sample_rate_hz, frequency_to_remove=notch_freq_hz)
        raise NotImplementedError()
        # TODO:
        # - check that power is within 95% rtol of expected
        # OR
        # - check that result exactly matches stored results array

    def test_bandpass(self):
        bp_low_hz = 2
        bp_high_hz = 45
        filtered_data = bandpass.butter_bandpass_filter(self.data, bp_low_hz, bp_high_hz, self.sample_rate_hz, order=2)
        raise NotImplementedError()
        # TODO:
        # - check that power is within 95% rtol of expected
        # OR
        # - check that result exactly matches stored results array


if __name__ == "__main__":
    unittest.main()
