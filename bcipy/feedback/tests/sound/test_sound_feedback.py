import unittest

import sounddevice as sd
from mockito import mock, unstub, when

from bcipy.feedback.sound.auditory_feedback import AuditoryFeedback


class TestSoundFeedback(unittest.TestCase):

    def setUp(self):
        """set up the needed path for load functions."""
        self.parameters = {}
        self.data_save_path = 'data/'
        self.user_information = 'test_user_0010'

        self.clock = mock()
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
        timestamp = 100
        when(self.clock).getTime().thenReturn(timestamp)
        resp = self.auditory_feedback.administer(
            self.sound, self.fs)

        self.assertTrue(isinstance(resp, list))
        self.assertEqual(resp[0], [self.auditory_feedback.feedback_timestamp_label, timestamp])


if __name__ == '__main__':
    unittest.main()
