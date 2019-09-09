from psychopy import core
import unittest

from mockito import mock, when, unstub

from bcipy.feedback.sound.auditory_feedback import AuditoryFeedback
from bcipy.helpers.load import load_json_parameters

import sounddevice as sd


class TestSoundFeedback(unittest.TestCase):

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters_used = 'bcipy/parameters/parameters.json'
        self.parameters = load_json_parameters(self.parameters_used)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_0010'

        self.clock = core.Clock()
        self.sound = mock()
        self.fs = mock()
        when(sd).play(self.sound, self.fs, blocking=True).thenReturn(None)

        self.auditory_feedback = AuditoryFeedback(
            parameters=self.parameters,
            clock=self.clock)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        unstub()

    def test_feedback_type(self):

        feedback_type = self.auditory_feedback._type()
        self.assertEqual(feedback_type, 'Auditory Feedback')

    def test_feedback_administer_sound(self):
        resp = self.auditory_feedback.administer(
            self.sound, self.fs)

        self.assertTrue(isinstance(resp, list))
