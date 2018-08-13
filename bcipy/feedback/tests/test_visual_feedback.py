import shutil
from psychopy import core
from psychopy import visual
import psychopy
import unittest

from mockito import any, mock, when, unstub

from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.load import load_json_parameters


class TestVisualFeedback(unittest.TestCase):

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters_used = 'bcipy/parameters/parameters.json'
        self.parameters = load_json_parameters(
            self.parameters_used,
            value_cast=True)

        self.display = mock()
        self.display.size = [1, 1]
        self.text_mock = mock()
        self.image_mock = mock()
        self.rect_mock = mock()

        self.clock = core.Clock()

        self.visual_feedback = VisualFeedback(
            display=self.display, parameters=self.parameters,
            clock=self.clock)

        when(psychopy.visual).TextStim(
            win=self.display,
            font=any(),
            text=any(),
            height=any(),
            pos=any(),
            color=any()).thenReturn(self.text_mock)

        when(psychopy.visual).TextStim(
            win=self.display,
            font=any(),
            text=any(),
            height=any(),
            pos=any()).thenReturn(self.text_mock)

        when(psychopy.visual).ImageStim(
            win=self.display,
            image=any(),
            mask=None,
            pos=any(),
            ori=any()
            ).thenReturn(self.image_mock)

        when(psychopy.visual).Rect(
            win=self.display,
            width=any(),
            height=any(),
            lineColor=any(),
            pos=any(),
            lineWidth=any(),
            ori=any()
            ).thenReturn(self.rect_mock)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        unstub()

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
        test_stimulus = 'bcipy/static/images/testing_images/white.png'
        resp = self.visual_feedback.administer(
            test_stimulus, message='Correct:')

        self.assertTrue(isinstance(resp, list))

    def test_feedback_assertion_images(self):
        test_stimulus = 'bcipy/static/images/testing_images/white.png'
        assertion = 'bcipy/static/images/testing_images/white.png'
        resp = self.visual_feedback.administer(
            test_stimulus, message='Correct:', compare_assertion=assertion)

        self.assertTrue(isinstance(resp, list))
