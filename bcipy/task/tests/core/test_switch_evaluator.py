"""Tests for Switch evidence evaluator"""

import unittest
from unittest.mock import Mock

import numpy as np

from bcipy.core.parameters import Parameters
from bcipy.task.control.evidence import SwitchEvaluator


class TestSwitchEvaluator(unittest.TestCase):
    """Tests for Switch evidence evaluator."""

    def setUp(self):
        """Setup common state"""
        self.symbol_set = []

        self.device_mock = Mock()
        self.device_mock.content_type = 'MARKERS'
        self.device_mock.channels = ['Marker']
        self.device_mock.sample_rate = 1

        self.transform_mock = Mock()
        self.signal_model_mock = Mock()
        meta_mock = Mock()
        meta_mock.device_spec = self.device_mock
        meta_mock.transform = self.transform_mock
        self.signal_model_mock.metadata = meta_mock
        self.signal_model_mock.compute_likelihood_ratio = Mock()

    def test_evaluate(self):
        """Test evaluation of evidence"""
        parameters = Parameters.from_cast_values(
            preview_inquiry_progress_method=1, stim_length=10)
        raw_data = np.array([1.0])

        # Parameters
        symbols = ["+", "H", "D", "J", "B", "C", "A", "F", "G", "I", "E"]
        times = [
            0.0, 0.56, 0.81, 1.08, 1.33, 1.58, 1.83, 2.08, 2.33, 2.58, 2.83
        ]
        target_info = ['nontarget'] * len(symbols)
        window_length = 0.5

        # Run the code
        evaluator = SwitchEvaluator(self.symbol_set, self.signal_model_mock,
                                    parameters)
        evaluator.evaluate(raw_data, symbols, times, target_info,
                           window_length)

        expected_reshaped_data = np.ones((1, 10, 1))
        reshaped = evaluator.preprocess(raw_data, times, target_info,
                                        window_length)
        # Assertions
        self.assertTrue(np.allclose(expected_reshaped_data, reshaped))
        self.signal_model_mock.compute_likelihood_ratio.assert_called_once()


if __name__ == '__main__':
    unittest.main()
