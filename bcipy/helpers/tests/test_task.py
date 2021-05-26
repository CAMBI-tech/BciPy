import unittest
import numpy as np
import psychopy
from collections import Counter
from mockito import unstub, mock, when, verifyStubbedInvocationsAreUsed

from bcipy.helpers.task import (
    alphabet,
    calculate_stimulation_freq,
    get_key_press,
    InquiryReshaper,
    TrialReshaper,
    _float_val,
    generate_targets,
    construct_triggers,
    target_info
)
from bcipy.helpers.load import load_json_parameters


class TestAlphabet(unittest.TestCase):
    def test_alphabet_text(self):
        parameters_used = './bcipy/parameters/parameters.json'
        parameters = load_json_parameters(parameters_used, value_cast=True)

        parameters['is_txt_stim'] = True

        alp = alphabet(parameters)

        self.assertEqual(
            alp,
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
             'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
             'Y', 'Z', '<',
             '_'])

    def test_alphabet_images(self):
        parameters_used = './bcipy/parameters/parameters.json'
        parameters = load_json_parameters(parameters_used, value_cast=True)

        parameters['is_txt_stim'] = False
        parameters['path_to_presentation_images'] = ('bcipy/static/images/'
                                                     'rsvp_images/')

        alp = alphabet(parameters)

        self.assertNotEqual(
            alp,
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
             'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<',
             '_'])


class TestTrialReshaper(unittest.TestCase):
    def setUp(self):
        self.target_info = ['first_pres_target', 'fixation',
                            'target', 'nontarget', 'nontarget']
        self.timing_info = [1.001, 1.2001, 1.4001, 1.6001, 1.8001]
        # make some fake eeg data
        self.channel_number = 21
        tmp_inp = np.array([range(4000)] * self.channel_number)
        # Add some channel info
        tmp_inp[:, 0] = np.transpose(np.arange(1, 22, 1))
        self.eeg = tmp_inp
        self.channel_map = [1] * self.channel_number

    def test_trial_reshaper(self):
        reshaped_trials, _ = TrialReshaper()(
            trial_labels=self.target_info,
            timing_info=self.timing_info, eeg_data=self.eeg,
            fs=256, channel_map=self.channel_map)

        self.assertTrue(
            len(reshaped_trials) == self.channel_number,
            f'len is {len(reshaped_trials)} not {self.channel_number}')
        self.assertEqual(len(reshaped_trials[0]), 3)


class TestInquiryReshaper(unittest.TestCase):
    def setUp(self):
        self.n_channel = 7
        self.trial_length = 0.5
        self.trials_per_inquiry = 3
        self.fs = 10
        self.target_info = [
            "first_pres_target", "fixation", "target", "nontarget", "nontarget",
            "first_pres_target", "fixation", "nontarget", "nontarget", "nontarget",
            "first_pres_target", "fixation", "nontarget", "target", "nontarget",
        ]
        self.timing_info = [
            1.0, 1.2, 1.4, 1.6, 1.8,
            2.0, 2.2, 2.4, 2.6, 2.8,
            3.0, 3.2, 3.4, 3.6, 3.8,
        ]

        # total duration = 4s
        self.eeg = np.random.randn(self.n_channel, int(self.fs * (4 + self.trial_length) * 2))
        self.channel_map = [1] * self.n_channel

    def test_inquiry_reshaper(self):
        reshaped_data, labels = InquiryReshaper()(
            trial_labels=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            fs=self.fs,
            trials_per_inquiry=self.trials_per_inquiry,
            channel_map=self.channel_map,
            trial_length=self.trial_length,
        )
        expected_shape = (self.n_channel, self.trials_per_inquiry, int(
            self.trial_length * self.fs) * self.trials_per_inquiry)
        self.assertTrue(reshaped_data.shape == expected_shape)
        self.assertTrue(all(labels == [0, 3, 1]))


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
        when(psychopy.event).getKeys(keyList=key_list, timeStamped=clock).thenReturn(key_response)

        # use the default label
        stamp_label = 'bcipy_key_press'
        expected = [f'{stamp_label}_{key_response[0][0]}', key_response[0][1]]
        response = get_key_press(key_list, clock)
        self.assertEqual(expected, response)

    def test_get_key_press_returns_none_if_no_keys_pressed(self):
        """Test for the case not keys are returned, ensures the calls occur with the correct inputs to psychopy"""

        key_list = ['space']
        key_response = None
        clock = mock()
        when(psychopy.event).getKeys(keyList=key_list, timeStamped=clock).thenReturn(key_response)

        response = get_key_press(key_list, clock)
        self.assertEqual(None, response)

    def test_get_key_press_set_custom_stamp_message(self):
        """Test for a custom stamp label, ensures the calls occur with the correct inputs to psychopy"""
        clock = mock()
        key_list = ['space']
        # get keys returns a list of lists with the key and timestamp per hit
        key_response = [[key_list[0], 1000]]
        when(psychopy.event).getKeys(keyList=key_list, timeStamped=clock).thenReturn(key_response)

        # set a custom label
        stamp_label = 'custom_label'
        expected = [f'{stamp_label}_{key_response[0][0]}', key_response[0][1]]
        response = get_key_press(key_list, clock, stamp_label=stamp_label)
        self.assertEqual(expected, response)


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
            'nontarget', 'nontarget', 'nontarget', 'target', 'nontarget',
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget'
        ]
        self.assertEqual(expected, target_info(triggers, target_letter='Z'))

        expected = [
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget'
        ]
        self.assertEqual(expected, target_info(triggers))
        self.assertEqual([], target_info([]))


if __name__ == '__main__':
    unittest.main()
