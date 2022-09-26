import unittest

from typing import List
from collections import Counter
from mockito import unstub, mock, when, verify, verifyStubbedInvocationsAreUsed

import numpy as np
import psychopy

from bcipy.acquisition import LslAcquisitionClient
from bcipy.acquisition.record import Record
from bcipy.task.exceptions import InsufficientDataException

from bcipy.helpers.task import (_float_val, alphabet,
                                calculate_stimulation_freq, construct_triggers,
                                generate_targets, get_data_for_decision,
                                get_key_press, target_info)


class TestAlphabet(unittest.TestCase):
    def test_alphabet_text(self):
        parameters = {}

        parameters['is_txt_stim'] = True

        alp = alphabet(parameters)

        self.assertEqual(alp, [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ])

    def test_alphabet_images(self):
        parameters = {}
        parameters['is_txt_stim'] = False
        parameters['path_to_presentation_images'] = ('bcipy/static/images/'
                                                     'rsvp/')

        alp = alphabet(parameters)

        self.assertNotEqual(alp, [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_'
        ])


class TestCalculateStimulationFreq(unittest.TestCase):
    def test_calculate_stimulate_frequency_returns_number_less_one(self):
        flash_time = 5
        stimulation_frequency = calculate_stimulation_freq(flash_time)
        expected = 1 / flash_time
        self.assertEqual(stimulation_frequency, expected)

    def test_calculate_stimulate_frequency_handles_zero(self):
        flash_time = 0
        with self.assertRaises(ZeroDivisionError):
            calculate_stimulation_freq(flash_time)


class TestFloatVal(unittest.TestCase):
    def test_float_val_as_str(self):
        col = 'Apple'
        result = _float_val(col)
        expected = 1.0
        self.assertEqual(result, expected)

    def test_float_val_as_int(self):
        col = 3
        result = _float_val(col)
        expected = 3.0
        self.assertEqual(result, expected)


class TestTargetGeneration(unittest.TestCase):
    """Tests for generation of target inquiries"""

    def test_target_number_less_than_alp(self):
        """Test when requested number of targets is less than the length of
        the alphabet."""
        alp = list(range(10))
        targets = generate_targets(alp, 5)
        self.assertEqual(len(targets), 5)
        self.assertEqual(len(targets), len(set(targets)))

    def test_target_greater_than_alp(self):
        """Test behavior when number of targets is greater than the length
        of the alphabet"""
        alp = list(range(5))
        targets = generate_targets(alp, 10)
        self.assertEqual(len(targets), 10)

        counts = Counter(targets)

        for item in alp:
            self.assertEqual(counts[item], 2)

    def test_remainder(self):
        """Test behavior when number of targets is greater than the length of
        the alphabet by a value other than a multiple of the alphabet length.
        """
        alp = list(range(5))
        targets = generate_targets(alp, 12)

        counts = Counter(targets)
        for item in alp:
            self.assertGreaterEqual(counts[item], 2)
            self.assertLessEqual(counts[item], 3)


class TestGetKeyPress(unittest.TestCase):
    """Tests for the get key press method"""

    def tearDown(self):
        verifyStubbedInvocationsAreUsed()
        unstub()

    def test_get_key_press_appends_stamp_label_defaults(self):
        """Test for the stamp label defaults, ensures the calls occur with the correct inputs to psychopy"""
        key_list = ['space']
        clock = mock()
        # get keys returns a list of lists with the key and timestamp per hit
        key_response = [[key_list[0], 1000]]
        when(psychopy.event).getKeys(keyList=key_list,
                                     timeStamped=True).thenReturn(key_response)
        when(clock).getTime().thenReturn(psychopy.core.getTime())
        # use the default label
        stamp_label = 'bcipy_key_press'
        expected = [f'{stamp_label}_{key_response[0][0]}', key_response[0][1]]
        response = get_key_press(key_list, clock)
        self.assertEqual(expected[0], response[0])
        self.assertAlmostEqual(expected[1], response[1], delta=0.01)

    def test_get_key_press_clock_adjustment(self):
        """Test for the stamp label defaults, ensures the calls occur with the correct inputs to psychopy"""
        key_list = ['space']
        clock = mock()
        # get keys returns a list of lists with the key and timestamp per hit
        key_response = [[key_list[0], 1000]]
        when(psychopy.event).getKeys(keyList=key_list,
                                     timeStamped=True).thenReturn(key_response)
        when(clock).getTime().thenReturn(psychopy.core.getTime() + 100)
        # use the default label
        stamp_label = 'bcipy_key_press'
        expected = [f'{stamp_label}_{key_response[0][0]}', key_response[0][1]]
        response = get_key_press(key_list, clock)
        self.assertEqual(expected[0], response[0])
        self.assertAlmostEqual(1100, response[1], delta=0.01)

    def test_get_key_press_returns_none_if_no_keys_pressed(self):
        """Test for the case not keys are returned, ensures the calls occur with the correct inputs to psychopy"""

        key_list = ['space']
        key_response = None
        clock = mock()
        when(psychopy.event).getKeys(keyList=key_list,
                                     timeStamped=True).thenReturn(key_response)

        response = get_key_press(key_list, clock)
        self.assertEqual(None, response)

    def test_get_key_press_set_custom_stamp_message(self):
        """Test for a custom stamp label, ensures the calls occur with the correct inputs to psychopy"""
        clock = mock()
        key_list = ['space']
        # get keys returns a list of lists with the key and timestamp per hit
        key_response = [[key_list[0], 1000]]
        when(psychopy.event).getKeys(keyList=key_list,
                                     timeStamped=True).thenReturn(key_response)
        when(clock).getTime().thenReturn(psychopy.core.getTime())
        # set a custom label
        stamp_label = 'custom_label'
        expected = [f'{stamp_label}_{key_response[0][0]}', key_response[0][1]]
        response = get_key_press(key_list, clock, stamp_label=stamp_label)
        self.assertEqual(expected[0], response[0])
        self.assertAlmostEqual(expected[1], response[1], delta=0.01)


class TestTriggers(unittest.TestCase):
    """Tests related to triggers"""

    def test_construct_triggers(self):
        stim_times = [['+', 7.009946188016329], ['<', 7.477798109990545],
                      ['_', 7.69470399999409], ['Z', 7.911495972017292],
                      ['U', 8.128477902995655], ['S', 8.345279764995212],
                      ['T', 8.562265532993479], ['V', 8.779025560012087],
                      ['X', 8.995945784990909], ['Y', 9.213076218002243],
                      ['W', 9.429835998016642]]
        expected = [('+', 0.0), ('<', 0.46785192197421566),
                    ('_', 0.6847578119777609), ('Z', 0.901549784000963),
                    ('U', 1.1185317149793264), ('S', 1.3353335769788828),
                    ('T', 1.5523193449771497), ('V', 1.7690793719957583),
                    ('X', 1.9859995969745796), ('Y', 2.203130029985914),
                    ('W', 2.4198898100003134)]
        self.assertEqual(expected, construct_triggers(stim_times))
        self.assertEqual([], construct_triggers([]))

    def test_target_info(self):
        triggers = [('+', 0.0), ('<', 0.46785192197421566),
                    ('_', 0.6847578119777609), ('Z', 0.901549784000963),
                    ('U', 1.1185317149793264), ('S', 1.3353335769788828),
                    ('T', 1.5523193449771497), ('V', 1.7690793719957583),
                    ('X', 1.9859995969745796), ('Y', 2.203130029985914),
                    ('W', 2.4198898100003134)]
        expected = [
            'fixation', 'nontarget', 'nontarget', 'target', 'nontarget',
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget'
        ]
        self.assertEqual(expected, target_info(triggers, target_letter='Z'))

        expected = [
            'fixation', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget'
        ]
        self.assertEqual(expected, target_info(triggers))
        self.assertEqual([], target_info([]))


def mock_get_data_response(samples: int, high: float, low: float,
                           channels: int) -> List[Record]:
    """Mock DataAcquisitionClient Response.

    The data acquisition client returns a list of records that need to be looped through
        to get the raw data without other items attached.
    """
    data = [np.random.uniform(low, high) for _ in range(channels)]
    record_data = []
    for i in range(samples):
        record_data.append(Record(data, i, None))
    return record_data


class TestGetDataForDecision(unittest.TestCase):
    def setUp(self) -> None:
        self.inquiry_timing = [('A', 1), ('B', 2), ('C', 3)]
        self.daq = mock(spec=LslAcquisitionClient)
        self.daq.device_spec = mock()
        self.daq.device_spec.sample_rate = 10
        self.mock_eeg = mock_get_data_response(samples=1000,
                                               high=1000,
                                               low=-1000,
                                               channels=4)

    def tearDown(self) -> None:
        unstub()

    def test_get_data_for_decision_returns_tuple_of_eeg_data_and_triggers(
            self):
        when(self.daq).get_data(start=any, limit=any).thenReturn(self.mock_eeg)

        response = get_data_for_decision(self.inquiry_timing, self.daq)

        self.assertIsInstance(response, tuple)

        _eeg_data, timing = response

        # self.assertEqual(eeg_data[:1][0], self.mock_eeg[0].data[0])
        self.assertIsInstance(timing, list)

    def test_get_data_for_decision_prestim(self):
        prestim = 1
        first_stim_time = self.inquiry_timing[0][1]
        last_stim_time = self.inquiry_timing[-1][1]

        expected_start = first_stim_time - prestim
        expected_stop = last_stim_time
        expected_triggers = [(text, ((timing - first_stim_time) + prestim))
                             for text, timing in self.inquiry_timing]
        expected_data_limit = round((expected_stop - expected_start) *
                                    self.daq.device_spec.sample_rate)

        when(self.daq).get_data(start=expected_start,
                                limit=expected_data_limit).thenReturn(
                                    self.mock_eeg)
        _, timing = get_data_for_decision(self.inquiry_timing,
                                          self.daq,
                                          prestim=prestim)

        # self.assertEqual(eeg_data[:1][0], self.mock_eeg[0].data[0])
        self.assertEqual(timing, expected_triggers)
        verify(self.daq, times=1).get_data(start=expected_start,
                                           limit=expected_data_limit)

    def test_get_data_for_decision_poststim(self):
        poststim = 1
        first_stim_time = self.inquiry_timing[0][1]
        last_stim_time = self.inquiry_timing[-1][1]

        expected_triggers = [(text, ((timing) - first_stim_time))
                             for text, timing in self.inquiry_timing]
        expected_data_limit = round(
            (last_stim_time - first_stim_time + poststim) *
            self.daq.device_spec.sample_rate)

        when(self.daq).get_data(start=first_stim_time,
                                limit=expected_data_limit).thenReturn(
                                    self.mock_eeg)
        _, timing = get_data_for_decision(self.inquiry_timing,
                                          self.daq,
                                          poststim=poststim)

        # self.assertEqual(eeg_data[:1][0], self.mock_eeg[0].data[0])
        self.assertEqual(timing, expected_triggers)
        verify(self.daq, times=1).get_data(start=first_stim_time,
                                           limit=expected_data_limit)

    def test_get_data_for_decision_offset(self):
        offset = 1
        first_stim_time = self.inquiry_timing[0][1]
        last_stim_time = self.inquiry_timing[-1][1]

        expected_start = first_stim_time + offset
        expected_stop = last_stim_time + offset
        expected_triggers = [(text, ((timing) - first_stim_time))
                             for text, timing in self.inquiry_timing]
        expected_data_limit = round((expected_stop - expected_start) *
                                    self.daq.device_spec.sample_rate)

        when(self.daq).get_data(start=expected_start,
                                limit=expected_data_limit).thenReturn(
                                    self.mock_eeg)
        _, timing = get_data_for_decision(self.inquiry_timing,
                                          self.daq,
                                          offset=offset)

        # self.assertEqual(eeg_data[:1][0], self.mock_eeg[0].data[0])
        self.assertEqual(timing, expected_triggers)
        verify(self.daq, times=1).get_data(start=expected_start,
                                           limit=expected_data_limit)

    def test_get_data_for_decision_throws_insufficient_data_error_if_less_than_data_limit(
            self):

        # return an empty list from the get data call
        when(self.daq).get_data(start=any, limit=any).thenReturn([])

        with self.assertRaises(InsufficientDataException):
            get_data_for_decision(self.inquiry_timing, self.daq)

    def test_get_data_for_decision_throws_insufficient_data_error_if_data_query_out_of_bounds(
            self):
        inquiry_timing = [('A', 10), ('D', 1)]

        with self.assertRaises(InsufficientDataException):
            get_data_for_decision(inquiry_timing, self.daq)


if __name__ == '__main__':
    unittest.main()
