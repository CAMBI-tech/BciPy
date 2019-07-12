import unittest
import numpy as np

from mockito import any, mock, when, unstub

from bcipy.helpers.task import alphabet, calculate_stimulation_freq, trial_reshaper, _float_val
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
        self.target_info = ['first_pres_target',
                            'target', 'nontarget', 'nontarget']
        self.timing_info = [1.001, 1.2001, 1.4001, 1.6001]
        # make some fake eeg data
        self.channel_number = 21
        tmp_inp = np.array([range(4000)] * self.channel_number)
        # Add some channel info
        tmp_inp[:, 0] = np.transpose(np.arange(1, 22, 1))
        self.eeg = tmp_inp
        self.channel_map = [1] * self.channel_number

    def tearDown(self):
        unstub()

    def test_trial_reshaper_calibration(self):
        (reshaped_trials, labels,
         num_of_seq, trials_per_sequence) = trial_reshaper(
            trial_target_info=self.target_info,
            timing_info=self.timing_info, eeg_data=self.eeg,
            fs=256, k=2, mode='calibration', channel_map=self.channel_map)

        self.assertTrue(
            len(reshaped_trials) == self.channel_number,
            f'len is {len(reshaped_trials)} not {self.channel_number}')
        self.assertEqual(len(reshaped_trials[0]), 3)
        self.assertEqual(num_of_seq, 1)
        self.assertEqual(trials_per_sequence, 3)

    def test_trial_reshaper_copy_phrase(self):
        pass


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


if __name__ == '__main__':
    unittest.main()
