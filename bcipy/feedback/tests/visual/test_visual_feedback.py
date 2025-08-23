import unittest

import psychopy
from mockito import (any, mock, unstub, verify, verifyNoUnwantedInteractions,
                     verifyStubbedInvocationsAreUsed, when)

from bcipy.feedback.visual.visual_feedback import FeedbackType, VisualFeedback
from bcipy.helpers.clock import Clock


class TestVisualFeedback(unittest.TestCase):

    def setUp(self):
        self.parameters = {
            'feedback_font': 'Arial',
            'feedback_stim_height': 1,
            'feedback_stim_width': 1,
            'feedback_color': 'white',
            'feedback_pos_x': 0,
            'feedback_pos_y': 0,
            'feedback_duration': 2,
            'feedback_line_width': 1
        }

        self.display = mock()
        self.text_mock = mock()
        self.image_mock = mock()

        self.clock = Clock()

        self.visual_feedback = VisualFeedback(
            display=self.display, parameters=self.parameters,
            clock=self.clock)

    def tearDown(self):
        verifyStubbedInvocationsAreUsed()
        verifyNoUnwantedInteractions()
        unstub()

    def test_feedback_type(self):
        feedback_type = self.visual_feedback._type()
        self.assertEqual(feedback_type, 'Visual Feedback')

    def test_construct_stimulus_image(self):
        image_mock = mock()
        image_mock.size = 0
        when(psychopy.visual).ImageStim(
            win=self.display,
            image=any(),
            mask=None,
            pos=any(),
            ori=any()
        ).thenReturn(image_mock)
        # mock the resize behavior for the image
        when(self.visual_feedback)._resize_image(
            any(), any(), any()).thenReturn()

        response = self.visual_feedback._construct_stimulus(
            'test_stim.png',
            (0, 0),
            None,
            FeedbackType.IMAGE,
        )

        self.assertEqual(response, image_mock)

    def test_construct_stimulus_text(self):
        text_mock = mock()
        stimulus = 'test'
        when(psychopy.visual).TextStim(
            win=self.display,
            font=self.visual_feedback.font_stim,
            text=stimulus,
            height=self.visual_feedback.height_stim,
            pos=any(),
            color=any()).thenReturn(text_mock)

        response = self.visual_feedback._construct_stimulus(
            stimulus,
            (0, 0),
            None,
            FeedbackType.TEXT,
        )

        self.assertEqual(response, text_mock)

    def test_show_stimuli(self):
        stimuli_mock = mock()
        when(stimuli_mock).draw().thenReturn(None)
        when(self.display).flip().thenReturn(None)

        response = self.visual_feedback._show_stimuli(
            stimuli_mock)  # TODO assertion

        verify(stimuli_mock, times=1).draw()
        verify(self.display, times=1).flip()

    def test_administer_default(self):
        stimulus = mock()
        timestamp = [self.visual_feedback.feedback_timestamp_label, 1000]
        when(self.visual_feedback)._construct_stimulus(
            stimulus,
            self.visual_feedback.pos_stim,
            self.visual_feedback.color,
            FeedbackType.TEXT
        ).thenReturn(stimulus)
        when(self.visual_feedback)._show_stimuli(
            stimulus).thenReturn(timestamp)
        when(psychopy.core).wait(
            self.visual_feedback.feedback_length).thenReturn()
        response = self.visual_feedback.administer(stimulus)
        expected = [timestamp]
        self.assertEqual(response, expected)


if __name__ == '__main__':
    unittest.main()
