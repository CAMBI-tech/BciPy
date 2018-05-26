import unittest
import sys

from bcipy.helpers.bci_task_related import alphabet, trial_reshaper
from bcipy.helpers.load import load_json_parameters


class TestAlphabet(unittest.TestCase):

    def test_alphabet_text(self):
        parameters_used = './parameters/parameters.json'
        parameters = load_json_parameters(parameters_used, value_cast=True)

        parameters['is_txt_sti'] = True
        parameters[
            'path_to_presentation_images'] = './bci/static/images/rsvp_images/'

        alp = alphabet(parameters)

        self.assertEqual(
            alp,
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
             'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
             'Y', 'Z', '<',
             '_'])

    def test_alphabet_images(self):
        parameters_used = './parameters/parameters.json'
        parameters = load_json_parameters(parameters_used, value_cast=True)

        parameters['is_txt_sti'] = False
        parameters['path_to_presentation_images'] = ('../bci/static/images/'
                                                     'rsvp_images/')

        alp = alphabet(parameters)

        self.assertNotEqual(
            alp,
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
             'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<',
             '_'])


class TestTrialReshaper(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trial_reshaper_calibration(self):
        pass

    def test_trial_reshaper_copy_phrase(self):
        pass
