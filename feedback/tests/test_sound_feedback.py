import shutil
from psychopy import core
import unittest

from feedback.sound.auditory_feedback import AuditoryFeedback
from helpers.load import load_json_parameters
from helpers.save import init_save_data_structure

import soundfile as sf


class TestSoundFeedback(unittest.TestCase):

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters_used = './parameters/parameters.json'
        self.parameters = load_json_parameters(self.parameters_used)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'

        self.save = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            self.parameters_used)

        self.clock = core.Clock()
        self.data, self.fs = sf.read(
            './static/sounds/1k_800mV_20ms_stereo.wav',
            dtype='float32')

        self.auditory_feedback = AuditoryFeedback(
            parameters=self.parameters,
            clock=self.clock)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.save)

    def test_feedback_type(self):

        feedback_type = self.auditory_feedback._type()
        self.assertEqual(feedback_type, 'Auditory Feedback')

    def test_feedback_administer_sound(self):
        resp = self.auditory_feedback.administer(
            self.sound, self.fs)

        self.assertTrue(isinstance(resp, list))
