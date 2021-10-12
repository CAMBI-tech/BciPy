from distutils.log import info
from typing import List, Tuple
import unittest
from bcipy.helpers.exceptions import BciPyCoreException

import psychopy
from mockito import (
    any,
    mock,
    verify,
    verifyNoMoreInteractions,
    when,
    unstub,
    verifyStubbedInvocationsAreUsed,
    verifyNoUnwantedInteractions
)
from bcipy.display import matrix
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
    task_color='W',
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
            wrapWidth=any(), colorSpace=any(),
            opacity=any(), depth=any()
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

    def test_schedule_to(self):
        #Test schedule_to method correctly sets stim inquiry, timing, and colors to given parameters.
        self.matrix.schedule_to(self.stimuli.stim_inquiry, self.stimuli.stim_timing, self.stimuli.stim_colors)

        self.assertEqual(self.matrix.stimuli_inquiry, self.stimuli.stim_inquiry)
        self.assertEqual(self.matrix.stimuli_timing, self.stimuli.stim_timing)
        self.assertEqual(self.matrix.stimuli_colors, self.stimuli.stim_colors)

    def test_do_inquiry_if_scp(self):
        # If SCP is true, matrix should animate.
        self.matrix.scp = True

        when(self.matrix).animate_scp().thenReturn()
        self.matrix.do_inquiry()
        verify(self.matrix, times=1).animate_scp()

    def test_do_inquiry_if_not_scp(self):
        # If SCP is false, an exception should be raised as currently there is no other inquiry option available.
        self.matrix.scp = False

        with self.assertRaises(BciPyCoreException):
            self.matrix.do_inquiry()

    def test_build_grid(self):
        # mock the text stims and increment_position
        when(psychopy.visual).TextStim(
            win=self.window,
            height=any(),
            text=any(),
            pos=any(),
            opacity=any()
        ).thenReturn(self.text_stim_mock)
        when(self.matrix).increment_position(any()).thenReturn()

        sym_length = len(self.matrix.symbol_set)    
        self.matrix.build_grid()
        # we expect the legth of the stimulus regisitry to be the same as the length of the symbol set
        self.assertEqual(len(self.matrix.stim_registry), sym_length)
        # verify that all of the text stims were drawn and the position was incremented each time
        verify(self.text_stim_mock, times=sym_length).draw()
        verify(self.matrix, times=sym_length).increment_position(any())

    def test_increment_position_increments_x_when_max_grid_width_not_met(self):
        self.matrix.position_increment = 0.2
        self.matrix.max_grid_width = 0.7
        self.matrix.position = (0, 0)
        # response should be 0,0 incremented by 0.2 in the x direction because 0.7 was not reached
        response = self.matrix.increment_position(self.matrix.position)
        self.assertEqual(response, (self.matrix.position_increment, 0))

    def test_increment_position_increments_y_when_max_grid_width_is_reached(self):
        self.matrix.position_increment = 0.2
        self.matrix.max_grid_width = 0.2
        self.matrix.position = (0, 0)
        # response should be 0,0 incremented by -0.2 in the y direction because 0.2 was reached
        response = self.matrix.increment_position(self.matrix.position)
        self.assertEqual(response, (0, -self.matrix.position_increment))

    def test_animate_scp(self):
        # mock the text stims
        when(psychopy.visual).TextStim(
            win=self.window,
            height=any(),
            text=any(),
            pos=any(),
            opacity=any()
        ).thenReturn(self.text_stim_mock)
        when(self.matrix.window).callOnFlip(any(), any(), any()).thenReturn()
        when(self.matrix.window).callOnFlip(any(), any()).thenReturn()
        # mock the drawing of text stims
        when(self.matrix).draw_static().thenReturn()
        when(self.text_stim_mock).draw().thenReturn()
        when(self.matrix.window).flip().thenReturn()
        # skip the core wait 
        when(psychopy.core).wait(any()).thenReturn()
        when(self.matrix.trigger_callback).reset().thenReturn()
        # we expect the timing returned to be a list
        response = self.matrix.animate_scp()
        self.assertIsInstance(response, list)

    def test_draw_static(self):
        info_text_mock = mock()
        # mock the task draw and info text draw methods
        when(self.matrix.task).draw().thenReturn()
        when(info_text_mock).draw().thenReturn()
        info_text_len = len(self.matrix.info_text)

        self.matrix.draw_static()
        # verify that task was drawn once and all info text ware drawn 
        verify(self.matrix.task, times=(info_text_len+1)).draw()  
        #verify(info_text_mock, times=info_text_len).draw()  

    def test_update_task(self):
        self.matrix.update_task(self.task_display.task_text, self.task_display.task_color, self.task_display.task_pos)
        # check the matrix task text, color, and position were updated
        self.assertEqual(self.task_display.task_text, self.matrix.task.text)
        self.assertEqual(self.task_display.task_color, self.matrix.task.color)
        self.assertEqual(self.task_display.task_pos, self.matrix.task.pos)

    def test_update_task_state(self):
        # mock update_task()
        when(self.matrix).update_task(self.task_display.task_text, self.task_display.task_color, self.task_display.task_pos).thenReturn()
        
        self.matrix.update_task_state(self.task_display.task_text, self.task_display.task_color)
        # verify that update_task() was called once
        verify(self.matrix, times=1).update_task(self.task_display.task_text, self.task_display.task_color, self.task_display.task_pos)

if __name__ == '__main__':
    unittest.main()
