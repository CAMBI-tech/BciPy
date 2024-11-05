import unittest
from unittest.mock import Mock, patch

import psychopy
from mockito import (mock, unstub, verify, verifyNoUnwantedInteractions,
                     verifyStubbedInvocationsAreUsed, when)

from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.button_press_handler import ButtonPressHandler
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.rsvp import RSVPDisplay

# Define some reusable elements to test RSVPDisplay with
LEN_STIM = 10
TEST_STIM = StimuliProperties(stim_font='Arial',
                              stim_pos=(0, 0),
                              stim_height=0.6,
                              stim_inquiry=['A', '+'] * LEN_STIM,
                              stim_colors=['white'] * LEN_STIM,
                              stim_timing=[3] * LEN_STIM,
                              is_txt_stim=True)
TEST_INFO = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Calibration Demo'],
)


class TestRSVPDisplay(unittest.TestCase):
    """This is Test Case for the RSVP Display"""

    @patch('bcipy.display.paradigm.rsvp.display.TaskBar')
    def setUp(self, task_bar_mock):
        """Set up needed items for test."""
        self.info = TEST_INFO
        self.stimuli = TEST_STIM
        self.window = mock({"units": "norm", "size": (2.0, 2.0)})
        self.experiment_clock = mock()
        self.static_clock = mock()
        self.text_stim_mock = mock()
        self.rect_stim_mock = mock()
        self.task_bar_mock = mock(TaskBar)
        task_bar_mock.returnValue(self.task_bar_mock)
        when(psychopy.visual).TextStim(...).thenReturn(self.text_stim_mock)
        self.rsvp = RSVPDisplay(self.window, self.static_clock,
                                self.experiment_clock, self.stimuli,
                                self.task_bar_mock, self.info)

    def tearDown(self):
        verifyNoUnwantedInteractions()
        verifyStubbedInvocationsAreUsed()
        unstub()

    def test_task_bar_config_properties_set_correctly(self):
        self.assertEqual(self.rsvp.task_bar, self.task_bar_mock)

    def test_information_properties_set_correctly(self):
        self.assertEqual(self.rsvp.info, self.info)
        self.assertEqual(self.rsvp.info_text,
                         self.info.build_info_text(self.window))

    def test_stimuli_properties_set_correctly(self):
        """Stimuli properties are set on the instance to allow easy resetting of this properties during a task."""
        self.assertEqual(self.rsvp.stimuli_inquiry, self.stimuli.stim_inquiry)
        self.assertEqual(self.rsvp.stimuli_colors, self.stimuli.stim_colors)
        self.assertEqual(self.rsvp.stimuli_timing, self.stimuli.stim_timing)
        self.assertEqual(self.rsvp.stimuli_font, self.stimuli.stim_font)
        self.assertEqual(self.rsvp.stimuli_height, self.stimuli.stim_height)
        self.assertEqual(self.rsvp.stimuli_pos, self.stimuli.stim_pos)
        self.assertEqual(self.rsvp.is_txt_stim, self.stimuli.is_txt_stim)
        self.assertEqual(self.rsvp.stim_length, self.stimuli.stim_length)


class TestRSVPDisplayInquiryPreview(unittest.TestCase):

    @patch('bcipy.display.paradigm.rsvp.display.init_preview_button_handler')
    @patch('bcipy.display.paradigm.rsvp.display.TaskBar')
    def setUp(self, task_bar_mock, init_preview_button_handler_mock):
        """Set up needed items for test."""
        self.info = TEST_INFO
        self.stimuli = TEST_STIM

        self.window = mock({"units": "norm", "size": (2.0, 2.0)})
        self.experiment_clock = mock()
        self.static_clock = mock()
        self.text_stim_mock = mock()
        self.rect_stim_mock = mock()
        self.task_bar_mock = mock(TaskBar)
        task_bar_mock.return_value(self.task_bar_mock)
        when(psychopy.visual).TextStim(...).thenReturn(self.text_stim_mock)

        self.rsvp = None

    def tearDown(self):
        verifyNoUnwantedInteractions()
        verifyStubbedInvocationsAreUsed()
        unstub()

    def init_display(self, preview_config) -> None:
        """Initializes a new display with the given config; """
        self.rsvp = RSVPDisplay(self.window,
                                self.static_clock,
                                self.experiment_clock,
                                self.stimuli,
                                self.task_bar_mock,
                                self.info,
                                preview_config=preview_config)

        # skip the gui components
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp).draw_preview().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(psychopy.core).wait(
            preview_config.preview_inquiry_isi).thenReturn()

    def test_preview_enabled_default(self):
        """Test preview_enabled when preview_config is None."""
        rsvp = RSVPDisplay(self.window,
                           self.static_clock,
                           self.experiment_clock,
                           self.stimuli,
                           self.task_bar_mock,
                           self.info,
                           preview_config=None)
        self.assertFalse(rsvp.preview_enabled)

    def test_preview_enabled_is_false(self):
        """Test inquiry preview disabled when params specify not to
        show_preview_inquiry."""
        preview_config = PreviewParams(show_preview_inquiry=False,
                                       preview_inquiry_length=2,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=1,
                                       preview_inquiry_isi=1,
                                       preview_box_text_size=0.1)
        rsvp = RSVPDisplay(self.window,
                           self.static_clock,
                           self.experiment_clock,
                           self.stimuli,
                           self.task_bar_mock,
                           self.info,
                           preview_config=preview_config)
        self.assertFalse(rsvp.preview_enabled)

    def test_preview_enabled_is_true(self):
        """Test inquiry preview disabled when params specify to
        show_preview_inquiry."""
        preview_config = PreviewParams(show_preview_inquiry=True,
                                       preview_inquiry_length=2,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=1,
                                       preview_inquiry_isi=1,
                                       preview_box_text_size=0.1)
        rsvp = RSVPDisplay(self.window,
                           self.static_clock,
                           self.experiment_clock,
                           self.stimuli,
                           self.task_bar_mock,
                           self.info,
                           preview_config=preview_config)
        self.assertTrue(rsvp.preview_enabled)

    @patch('bcipy.display.paradigm.rsvp.display.init_preview_button_handler')
    def test_preview_inquiry_evoked_press_to_accept_pressed(
            self, init_preview_button_handler_mock):
        """Test preview with a press to accept configuration."""

        button_press_handler_mock = Mock(ButtonPressHandler)
        init_preview_button_handler_mock.return_value = button_press_handler_mock

        preview_config = PreviewParams(show_preview_inquiry=True,
                                       preview_inquiry_length=0.01,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=1,
                                       preview_inquiry_isi=0.01,
                                       preview_box_text_size=0.1)
        self.init_display(preview_config)

        button_press_handler_mock.has_response.return_value = True
        button_press_handler_mock.accept_result.return_value = True
        button_press_handler_mock.response_label = 'bcipy_key_press_space'
        button_press_handler_mock.response_timestamp = 1.0

        timing = []
        proceed = self.rsvp.preview_inquiry(timing)

        init_preview_button_handler_mock.assert_called_once_with(
            preview_config, self.experiment_clock)
        button_press_handler_mock.await_response.assert_called_once()

        verify(self.rsvp, times=1).draw_preview()
        self.assertTrue(proceed)
        self.assertEqual(
            2, len(timing),
            "time stamps for the preview and the button press should be present"
        )
        label, stamp = timing[-1]
        self.assertEqual('bcipy_key_press_space', label)
        self.assertEqual(1.0, stamp)

    @patch('bcipy.display.paradigm.rsvp.display.init_preview_button_handler')
    def test_preview_inquiry_evoked_press_to_skip_pressed(
            self, init_preview_button_handler_mock):
        """Test preview with a press to reject/skip configuration."""
        button_press_handler_mock = Mock(ButtonPressHandler)
        init_preview_button_handler_mock.return_value = button_press_handler_mock

        preview_config = PreviewParams(show_preview_inquiry=True,
                                       preview_inquiry_length=0.01,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=2,
                                       preview_inquiry_isi=0.01,
                                       preview_box_text_size=0.1)
        self.init_display(preview_config)

        button_press_handler_mock.has_response.return_value = True
        button_press_handler_mock.accept_result.return_value = False
        button_press_handler_mock.response_label = 'bcipy_key_press_space'
        button_press_handler_mock.response_timestamp = 1.0

        timing = []
        proceed = self.rsvp.preview_inquiry(timing)

        verify(self.rsvp, times=1).draw_preview()
        self.assertFalse(proceed)
        self.assertEqual(2, len(timing))
        label, stamp = timing[-1]
        self.assertEqual('bcipy_key_press_space', label)
        self.assertEqual(1.0, stamp)

    @patch('bcipy.display.paradigm.rsvp.display.init_preview_button_handler')
    def test_preview_inquiry_evoked_press_to_accept_not_pressed(
            self, init_preview_button_handler_mock):
        """Test preview with a press to accept with no button press."""
        button_press_handler_mock = Mock(ButtonPressHandler)
        init_preview_button_handler_mock.return_value = button_press_handler_mock

        preview_config = PreviewParams(show_preview_inquiry=True,
                                       preview_inquiry_length=0.01,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=1,
                                       preview_inquiry_isi=0.01,
                                       preview_box_text_size=0.1)
        self.init_display(preview_config)

        button_press_handler_mock.has_response.return_value = False
        button_press_handler_mock.accept_result.return_value = False

        timing = []
        proceed = self.rsvp.preview_inquiry(timing)
        verify(self.rsvp, times=1).draw_preview()

        self.assertFalse(proceed)
        self.assertEqual(1, len(timing),
                         "Only the timing for the preview should be present.")

    @patch('bcipy.display.paradigm.rsvp.display.init_preview_button_handler')
    def test_preview_inquiry_evoked_press_to_skip_not_pressed(
            self, init_preview_button_handler_mock):
        """Test preview with a press to reject/skip configuration when the
        button was not pressed."""
        button_press_handler_mock = Mock(ButtonPressHandler)
        init_preview_button_handler_mock.return_value = button_press_handler_mock

        preview_config = PreviewParams(show_preview_inquiry=True,
                                       preview_inquiry_length=0.01,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=2,
                                       preview_inquiry_isi=0.01,
                                       preview_box_text_size=0.1)
        self.init_display(preview_config)

        button_press_handler_mock.has_response.return_value = False
        button_press_handler_mock.accept_result.return_value = True

        timing = []
        proceed = self.rsvp.preview_inquiry(timing)

        verify(self.rsvp, times=1).draw_preview()

        self.assertTrue(proceed)
        self.assertEqual(1, len(timing),
                         "Only the timing for the preview should be present.")

    @patch('bcipy.display.paradigm.rsvp.display.init_preview_button_handler')
    def test_preview_inquiry_preview_only_response_registered(
            self, init_preview_button_handler_mock):
        """Test preview with preview only."""
        button_press_handler_mock = Mock(ButtonPressHandler)
        init_preview_button_handler_mock.return_value = button_press_handler_mock

        preview_config = PreviewParams(show_preview_inquiry=True,
                                       preview_inquiry_length=0.01,
                                       preview_inquiry_key_input='space',
                                       preview_inquiry_progress_method=0,
                                       preview_inquiry_isi=0.01,
                                       preview_box_text_size=0.1)
        self.init_display(preview_config)

        button_press_handler_mock.has_response.return_value = False
        button_press_handler_mock.accept_result.return_value = True

        timing = []
        proceed = self.rsvp.preview_inquiry(timing)

        verify(self.rsvp, times=1).draw_preview()
        self.assertTrue(proceed)


if __name__ == '__main__':
    unittest.main()
