import shutil
from psychopy import core
import unittest

from feedback.visual.visual_feedback import VisualFeedback
from helpers.load import load_json_parameters
from helpers.save import init_save_data_structure

from display.display_main import init_display_window


class TestVisualFeedback(unittest.TestCase):

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters_used = './parameters/parameters.json'
        self.parameters = load_json_parameters(
            self.parameters_used,
            value_cast=True)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_005'

        self.save = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            self.parameters_used)

        self.display = init_display_window(self.parameters)

        self.clock = core.Clock()

        self.visual_feedback = VisualFeedback(
            display=self.display, parameters=self.parameters,
            clock=self.clock)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.data_save_path)

    def test_feedback_type(self):

        feedback_type = self.visual_feedback._type()
        self.assertEqual(feedback_type, 'Visual Feedback')

    def test_feedback_administer_text(self):
        test_stimulus = 'A'
        resp = self.visual_feedback.administer(
            test_stimulus, message='Correct:')

        self.assertTrue(isinstance(resp, list))

    def test_feedback_assertion_text(self):
        stimulus = 'B'
        assertion = 'A'
        self.visual_feedback.message_color = 'red'
        resp = self.visual_feedback.administer(
            stimulus,
            message='Incorrect:', compare_assertion=assertion)
        self.assertTrue(isinstance(resp, list))

    def test_feedback_administer_image(self):
        test_stimulus = './static/images/testing_images/white.png'
        resp = self.visual_feedback.administer(
            test_stimulus, message='Correct:')

        self.assertTrue(isinstance(resp, list))

    def test_feedback_assertion_images(self):
        test_stimulus = './static/images/testing_images/white.png'
        assertion = './static/images/testing_images/white.png'
        resp = self.visual_feedback.administer(
            test_stimulus, message='Correct:', compare_assertion=assertion)

        self.assertTrue(isinstance(resp, list))
