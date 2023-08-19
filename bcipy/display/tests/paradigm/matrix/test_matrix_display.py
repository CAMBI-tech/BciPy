"""Tests for MatrixDisplay"""
import unittest

import psychopy
from mockito import (any, mock, unstub, verify, verifyNoUnwantedInteractions,
                     when)

from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay, SymbolDuration

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

TEST_INFO = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Matrix Calibration Demo'],
)


class TestMatrixDisplay(unittest.TestCase):
    """This is Test Case for the Matrix Display"""

    def setUp(self):
        """Set up needed items for test."""
        self.info = TEST_INFO
        self.stimuli = TEST_STIM
        self.window = mock({"units": "norm", "size": (2.0, 2.0)})
        self.experiment_clock = mock()
        self.static_clock = mock()
        self.text_stim_mock = mock(psychopy.visual.TextStim)
        when(self.text_stim_mock).setOpacity(...).thenReturn()
        when(self.text_stim_mock).setColor(...).thenReturn()
        when(self.text_stim_mock).draw(...).thenReturn()
        # grid item
        when(psychopy.visual).TextStim(
            win=self.window,
            text=any(),
            color=any(),
            opacity=any(),
            pos=any(),
            height=any(),
        ).thenReturn(self.text_stim_mock)

        # target
        when(psychopy.visual).TextStim(win=self.window,
                                       font=any(),
                                       text=any(),
                                       height=any(),
                                       color=any(),
                                       wrapWidth=any()).thenReturn(
                                           self.text_stim_mock)

        # waitscreen
        when(psychopy.visual).TextStim(win=self.window,
                                       font=any(),
                                       text=any(),
                                       height=any(),
                                       color=any(),
                                       pos=any(),
                                       wrapWidth=any()).thenReturn(
                                           self.text_stim_mock)
        # stim make functions
        when(psychopy.visual).TextStim(win=self.window,
                                       color=any(),
                                       height=any(),
                                       text=any(),
                                       font=any(),
                                       pos=any(),
                                       wrapWidth=any(),
                                       colorSpace=any(),
                                       opacity=any(),
                                       depth=any()).thenReturn(
                                           self.text_stim_mock)
        when(psychopy.visual).TextStim(...).thenReturn(self.text_stim_mock)
        self.task_bar_mock = mock(TaskBar)
        self.matrix = MatrixDisplay(window=self.window,
                                    experiment_clock=self.experiment_clock,
                                    stimuli=self.stimuli,
                                    task_bar=self.task_bar_mock,
                                    info=self.info)
        when(self.matrix)._trigger_pulse().thenReturn()

    def tearDown(self):
        verifyNoUnwantedInteractions()
        # verifyStubbedInvocationsAreUsed()
        unstub()

    def test_information_properties_set_correctly(self):
        self.assertEqual(self.matrix.info_text, self.info.build_info_text(self.window))

    def test_schedule_to(self):
        # Test schedule_to method correctly sets stim inquiry, timing, and colors to given parameters.
        self.matrix.schedule_to(self.stimuli.stim_inquiry, self.stimuli.stim_timing, self.stimuli.stim_colors)

        self.assertEqual(self.matrix.stimuli_inquiry, self.stimuli.stim_inquiry)
        self.assertEqual(self.matrix.stimuli_timing, self.stimuli.stim_timing)

    def test_do_inquiry(self):
        stimuli = ['X', '+', 'A', 'B', 'C']
        timing = [0.5, 1.0, 0.2, 0.2, 0.2]

        target = SymbolDuration('X', 0.5)
        fixation = SymbolDuration('+', 1.0)
        stim_durations = [
            SymbolDuration(*sti)
            for sti in [('A', 0.2), ('B', 0.2), ('C', 0.2)]
        ]

        when(self.matrix).prompt_target(any()).thenReturn(0.0)
        when(self.matrix).animate_scp(any(), any()).thenReturn([])

        self.matrix.schedule_to(stimuli, timing, [])
        self.matrix.do_inquiry()

        verify(self.matrix, times=1)._trigger_pulse()
        verify(self.matrix, times=1).prompt_target(target)
        verify(self.matrix, times=1).animate_scp(fixation, stim_durations)

    def test_build_grid(self):
        # mock the text stims and increment_position
        when(psychopy.visual).TextStim(
            win=self.window,
            height=any(),
            text=any(),
            color=any(),
            pos=any(),
            opacity=any()
        ).thenReturn(self.text_stim_mock)
        when(self.matrix).increment_position(any()).thenReturn()

        sym_length = len(self.matrix.symbol_set)
        grid = self.matrix.build_grid()

        self.assertEqual(len(grid), sym_length)
        # verify that the position was incremented each time
        verify(self.matrix, times=sym_length).increment_position(any())

    def test_draw_grid(self):
        """Test that all items in the grid draw."""
        self.matrix.draw_grid()
        verify(self.text_stim_mock, times=len(self.matrix.symbol_set)).draw()


    def test_animate_scp(self):
        # mock the text stims and window
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

        self.matrix.animate_scp(
            fixation=SymbolDuration('+', 1),
            stimuli=[SymbolDuration('A', 1),
                     SymbolDuration('Z', 1)])
        verify(self.matrix.window, times=1).callOnFlip(any(), '+')
        verify(self.matrix.window, times=1).callOnFlip(any(), 'A')
        verify(self.matrix.window, times=1).callOnFlip(any(), 'Z')

    def test_draw_static(self):
        # mock the task draw and info text draw methods
        when(self.matrix.task_bar).draw().thenReturn()
        info_mock = mock()
        self.matrix.info_text = [info_mock]
        when(info_mock).draw().thenReturn()
        info_text_len = len(self.matrix.info_text)

        self.matrix.draw_static()
        # verify that task was drawn once and all info text ware drawn
        verify(self.matrix.task_bar, times=1).draw()
        verify(info_mock, times=info_text_len).draw()


if __name__ == '__main__':
    unittest.main()
