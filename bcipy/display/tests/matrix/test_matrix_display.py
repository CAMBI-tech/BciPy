import unittest

import psychopy
from mockito import (
    any,
    mock,
    when,
    unstub,
    verifyStubbedInvocationsAreUsed,
    verifyNoUnwantedInteractions
)
from bcipy.display.matrix import (
    MatrixDisplay
)
from bcipy.display import (
    StimuliProperties,
    InformationProperties,
    TaskDisplayProperties,
)

# Define some reusable elements to test Matrix Display with
LEN_STIM = 10
TEST_STIM = StimuliProperties(
    stim_font='Arial',
    stim_pos=(-0.6, 0.4),
    stim_height=0.1,
    stim_inquiry=['A'],
    stim_colors=[],
    stim_timing=[0.1],
    is_txt_stim=True)
TEST_TASK_DISPLAY = TaskDisplayProperties(
    task_color=['White'],
    task_pos=(-.8, .85),
    task_font='Arial',
    task_height=.1,
    task_text='1/100'
)
TEST_INFO = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Matrix Calibration Demo'],
)


class TestMatrixDisplay(unittest.TestCase):
    """This is Test Case for the Matrix Display"""
    # TODO: Set these up for Matrix Display!

    def setUp(self):
        """Set up needed items for test."""
        self.info = TEST_INFO
        self.task_display = TEST_TASK_DISPLAY
        self.stimuli = TEST_STIM
        self.window = mock()
        self.experiment_clock = mock()
        self.static_clock = mock()
        self.text_stim_mock = mock()
        when(psychopy.visual).TextStim(
            win=self.window,
            color=any(),
            height=any(),
            text=any(),
            font=any(),
            pos=any(),
            wrapWidth=None, colorSpace='rgb',
            opacity=1, depth=-6.0
        ).thenReturn(self.text_stim_mock)
        self.matrix = MatrixDisplay(
            self.window,
            self.static_clock,
            self.experiment_clock,
            self.stimuli,
            self.task_display,
            self.info)

    def tearDown(self):
        # verifyNoUnwantedInteractions()
        # verifyStubbedInvocationsAreUsed()
        unstub()

    def test_task_display_properties_set_correctly(self):
        self.assertEqual(self.matrix.task_display, self.task_display)
        self.assertEqual(self.matrix.task, self.task_display.build_task(self.window))

    def test_information_properties_set_correctly(self):
        self.assertEqual(self.matrix.info, self.info)
        self.assertEqual(self.matrix.info_text, self.info.build_info_text(self.window))

    def test_stimuli_properties_set_correctly(self):
        """Stimuli properties are set on the instance to allow easy resetting of this properties during a task."""
        self.assertEqual(self.matrix.stimuli_inquiry, self.stimuli.stim_inquiry)
        self.assertEqual(self.matrix.stimuli_colors, self.stimuli.stim_colors)
        self.assertEqual(self.matrix.stimuli_timing, self.stimuli.stim_timing)
        self.assertEqual(self.matrix.stimuli_font, self.stimuli.stim_font)
        self.assertEqual(self.matrix.stimuli_height, self.stimuli.stim_height)
        self.assertEqual(self.matrix.stimuli_pos, self.stimuli.stim_pos)
        self.assertEqual(self.matrix.is_txt_stim, self.stimuli.is_txt_stim)
        self.assertEqual(self.matrix.stim_length, self.stimuli.stim_length)

    def test_increment_position_increments_x_when_max_grid_width_not_met(self):
        self.matrix.position_increment = 0.2
        self.matrix.max_grid_width = 0.7
        self.matrix.position = (0, 0)

        response = self.matrix.increment_position(self.matrix.position)
        self.assertEqual(response, (self.matrix.position_increment, 0))

    def test_increment_position_increments_y_when_max_grid_width_is_reached(self):
        self.matrix.position_increment = 0.2
        self.matrix.max_grid_width = 0.2
        self.matrix.position = (0, 0)

        response = self.matrix.increment_position(self.matrix.position)
        self.assertEqual(response, (0, -self.matrix.position_increment))


if __name__ == '__main__':
    unittest.main()
