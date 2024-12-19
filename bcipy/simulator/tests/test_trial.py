"""Test for trial data"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from bcipy.simulator.data.data_process import (DecodedTriggers,
                                               ExtractedExperimentData)
from bcipy.simulator.data.trial import (Trial, convert_trials, series_inquiry,
                                        session_series_counts)


class TestTrial(unittest.TestCase):
    """Test for trial class and functions."""

    def test_trial_str(self):
        """Test representation of a trial"""
        trial = Trial(source="path-to-source",
                      series=1,
                      series_inquiry=2,
                      inquiry_n=2,
                      inquiry_pos=5,
                      symbol='M',
                      target=1,
                      eeg=np.zeros(shape=(3, 2)))
        self.assertTrue('series=1' in str(trial))
        self.assertTrue('eeg=(3, 2)' in str(trial))

    def test_trial_without_optional(self):
        """Test without the optional data"""
        trial = Trial(source="path-to-source",
                      series=None,
                      series_inquiry=None,
                      inquiry_n=2,
                      inquiry_pos=5,
                      symbol='M',
                      target=1,
                      eeg=np.zeros(shape=(3, 2)))
        self.assertTrue('series=None' in str(trial))

    @patch("bcipy.simulator.data.trial.read_session")
    @patch("bcipy.simulator.data.trial.Path")
    def test_session_series_counts(self, path_constructor_mock,
                                   read_session_mock):
        """Test computing the series counts from the session data when the
        session path exists."""
        path_mock = Mock()
        path_mock.exists.return_value = True
        path_constructor_mock.return_value = path_mock
        session_series_counts("example-data-directory")
        read_session_mock.assert_called_once()

    @patch("bcipy.simulator.data.trial.read_session")
    @patch("bcipy.simulator.data.trial.Path")
    def test_session_series_counts_bad_path(self, path_constructor_mock,
                                            read_session_mock):
        """Test computing the series counts from the session data when session
        data is not present."""
        path_mock = Mock()
        path_mock.exists.return_value = False
        path_constructor_mock.return_value = path_mock
        session_series_counts("no-session")
        read_session_mock.assert_not_called()

    def test_series_inquiry(self):
        """Test series inquiry calculation"""
        counts = [5, 2, 4]
        self.assertEqual((1, 0), series_inquiry(counts, 0))
        self.assertEqual((1, 4), series_inquiry(counts, 4))
        self.assertEqual((2, 0), series_inquiry(counts, 5))
        self.assertEqual((2, 1), series_inquiry(counts, 6))
        self.assertEqual((3, 0), series_inquiry(counts, 7))
        self.assertEqual((3, 2), series_inquiry(counts, 9))
        self.assertEqual((3, 3), series_inquiry(counts, 10))

        self.assertEqual((None, None), series_inquiry([], 0))

    def test_convert_trials(self):
        """Test converting data to trials."""

        sample_data = ExtractedExperimentData(
            source_dir="test-src",
            inquiries=np.ones((6, 4, 436)),
            trials=np.ones((6, 12, 75)),
            labels=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
            inquiry_timing=[[150, 180, 210], [150, 180, 210], [150, 180, 210],
                            [150, 180, 210]],
            decoded_triggers=DecodedTriggers(
                targetness=[
                    'nontarget', 'nontarget', 'nontarget', 'nontarget',
                    'nontarget', 'nontarget', 'nontarget', 'nontarget',
                    'nontarget', 'nontarget', 'nontarget', 'target'
                ],
                times=[
                    10.515330292284489, 10.717246917076409, 10.919030375313014,
                    15.644844584167004, 15.846348999999464, 16.047540083993226,
                    20.790782417170703, 20.992313000373542, 21.193465375341475,
                    25.914405125193298, 26.115663417149335, 26.31681716721505
                ],
                symbols=[
                    '<', 'N', 'J', '<', 'N', 'J', 'J', 'B', '<', '<', 'J', 'B'
                ],
                corrected_times=[
                    10.515330292284489, 10.717246917076409, 10.919030375313014,
                    15.644844584167004, 15.846348999999464, 16.047540083993226,
                    20.790782417170703, 20.992313000373542, 21.193465375341475,
                    25.914405125193298, 26.115663417149335, 26.31681716721505
                ]),
            trials_per_inquiry=3)

        trials = convert_trials(sample_data, get_series_counts=lambda x: [4])
        self.assertEqual(12, len(trials))

        # check properties for first element
        self.assertEqual(trials[0].source, "test-src")
        self.assertEqual(trials[0].series, 1)
        self.assertEqual(trials[0].series_inquiry, 0)
        self.assertEqual(trials[0].inquiry_n, 0)
        self.assertEqual(trials[0].inquiry_pos, 1)
        self.assertEqual(trials[0].symbol, '<')
        self.assertEqual(trials[0].target, 0)
        self.assertEqual(trials[0].eeg.shape, (6, 75))

        # check properties for last element
        self.assertEqual(trials[-1].source, "test-src")
        self.assertEqual(trials[-1].series, 1)
        self.assertEqual(trials[-1].series_inquiry, 3)
        self.assertEqual(trials[-1].inquiry_n, 3)
        self.assertEqual(trials[-1].inquiry_pos, 3)
        self.assertEqual(trials[-1].symbol, 'B')
        self.assertEqual(trials[-1].target, 1)
        self.assertEqual(trials[-1].eeg.shape, (6, 75))

        # check properties when series_counts are different
        trials = convert_trials(sample_data,
                                get_series_counts=lambda x: [3, 1])
        self.assertEqual(trials[-1].series, 2)
        self.assertEqual(trials[-1].series_inquiry, 0)
        self.assertEqual(trials[-1].inquiry_n, 3)
        self.assertEqual(trials[-1].inquiry_pos, 3)
        self.assertEqual(trials[-1].symbol, 'B')
        self.assertEqual(trials[-1].target, 1)
        self.assertEqual(trials[-1].eeg.shape, (6, 75))


if __name__ == '__main__':
    unittest.main()
