"""Tests for EEG evidence evaluator"""

import unittest
from unittest.mock import Mock, patch

from bcipy.task.control.evidence import EegEvaluator, FilterParams


class TestEegEvaluator(unittest.TestCase):
    """Tests for EEG evidence evaluator."""

    def setUp(self):
        """Setup common state"""
        self.symbol_set = []

        self.device_mock = Mock()
        self.device_mock.content_type = 'EEG'
        self.device_mock.channels = ['ch1', 'ch2', 'ch3']
        self.device_mock.sample_rate = 300

        self.filter_params = FilterParams()
        self.signal_model_mock = Mock()

    @patch('bcipy.task.control.evidence.analysis_channels')
    def test_init(self, analysis_channels_mock):
        """Test initialization"""
        EegEvaluator(self.symbol_set, self.signal_model_mock, self.device_mock,
                     self.filter_params)
        analysis_channels_mock.assert_called_with(self.device_mock.channels,
                                                  self.device_mock)

    @patch('bcipy.task.control.evidence.analysis_channels')
    @patch('bcipy.task.control.evidence.get_default_transform')
    @patch('bcipy.task.control.evidence.TrialReshaper')
    def test_evaluate(self, reshaper_factory_mock, transform_factory_mock,
                      analysis_channels_mock):
        """Test evaluation of evidence"""

        # Mocks
        raw_data = Mock()
        channel_map = Mock()
        analysis_channels_mock.return_value = channel_map

        transformed_data = Mock()
        transformer_mock = Mock()
        transform_factory_mock.return_value = transformer_mock
        transformer_mock.return_value = (transformed_data, 150)

        reshaper_mock = Mock()
        reshaper_factory_mock.return_value = reshaper_mock
        reshaped_data = Mock()
        reshaper_mock.return_value = reshaped_data, Mock()

        # Parameters
        symbols = ["+", "H", "D", "J", "B", "C", "A", "F", "G", "I", "E"]
        times = [
            0.0, 0.56, 0.81, 1.08, 1.33, 1.58, 1.83, 2.08, 2.33, 2.58, 2.83
        ]
        target_info = ['nontarget'] * len(symbols)
        window_length = 0.5

        # Run the code
        evaluator = EegEvaluator(self.symbol_set, self.signal_model_mock,
                                 self.device_mock, self.filter_params)
        evaluator.evaluate(raw_data, symbols, times, target_info,
                           window_length)

        # Assertions
        transform_factory_mock.assert_called_once_with(
            sample_rate_hz=self.device_mock.sample_rate,
            notch_freq_hz=self.filter_params.notch_filter_frequency,
            bandpass_low=self.filter_params.filter_low,
            bandpass_high=self.filter_params.filter_high,
            bandpass_order=self.filter_params.filter_order,
            downsample_factor=self.filter_params.downsample_rate)

        transformer_mock.assert_called_once_with(raw_data,
                                                 self.device_mock.sample_rate)

        reshaper_mock.assert_called_once_with(
            trial_targetness_label=target_info,
            timing_info=times,
            eeg_data=transformed_data,
            sample_rate=150,
            channel_map=channel_map,
            poststimulus_length=window_length)

        self.signal_model_mock.predict.assert_called_once_with(
            reshaped_data, symbols, self.symbol_set)


if __name__ == '__main__':
    unittest.main()
