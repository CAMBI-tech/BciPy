import unittest

import psychopy
from mock import patch
from mockito import (
    any,
    mock,
    when,
    unstub,
    verifyStubbedInvocationsAreUsed,
    verifyNoUnwantedInteractions
)
from bcipy.display.paradigm.rsvp import (
    RSVPDisplay
)
from bcipy.display import (
    StimuliProperties,
    InformationProperties,
    TaskDisplayProperties,
    PreviewInquiryProperties,
)

# Define some reusable elements to test RSVPDisplay with
LEN_STIM = 10
TEST_STIM = StimuliProperties(
    stim_font='Arial',
    stim_pos=(0, 0),
    stim_height=0.6,
    stim_inquiry=['A', '+'] * LEN_STIM,
    stim_colors=['white'] * LEN_STIM,
    stim_timing=[3] * LEN_STIM,
    is_txt_stim=True)
TEST_TASK_DISPLAY = TaskDisplayProperties(
    task_color=['White'],
    task_pos=(-.5, .8),
    task_font='Arial',
    task_height=.1,
    task_text='1/100'
)
TEST_INFO = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Calibration Demo'],
)


class TestRSVPDisplay(unittest.TestCase):
    """This is Test Case for the RSVP Display"""

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
        self.rsvp = RSVPDisplay(
            self.window,
            self.static_clock,
            self.experiment_clock,
            self.stimuli,
            self.task_display,
            self.info)

    def tearDown(self):
        verifyNoUnwantedInteractions()
        verifyStubbedInvocationsAreUsed()
        unstub()

    def test_task_display_properties_set_correctly(self):
        self.assertEqual(self.rsvp.task_display, self.task_display)
        self.assertEqual(self.rsvp.task, self.task_display.build_task(self.window))

    def test_information_properties_set_correctly(self):
        self.assertEqual(self.rsvp.info, self.info)
        self.assertEqual(self.rsvp.info_text, self.info.build_info_text(self.window))

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
    def setUp(self):
        """Set up needed items for test."""
        self.info = TEST_INFO
        self.task_display = TEST_TASK_DISPLAY
        self.stimuli = TEST_STIM
        self.preview_inquiry_length = 0.1
        self.preview_inquiry_isi = 0.1
        self.preview_inquiry_progress_method = 1  # preview only = 0; press to accept == 1; press to skip == 2
        self.preview_inquiry_key_input = 'space'
        self.preview_inquiry = PreviewInquiryProperties(
            preview_only=False,
            preview_inquiry_length=self.preview_inquiry_length,
            preview_inquiry_isi=self.preview_inquiry_isi,
            preview_inquiry_progress_method=self.preview_inquiry_progress_method,
            preview_inquiry_key_input=self.preview_inquiry_key_input
        )
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
        self.rsvp = RSVPDisplay(
            self.window,
            self.static_clock,
            self.experiment_clock,
            self.stimuli,
            self.task_display,
            self.info,
            preview_inquiry=self.preview_inquiry)

    def tearDown(self):
        verifyNoUnwantedInteractions()
        verifyStubbedInvocationsAreUsed()
        unstub()

    def test_preview_inquiry_properties_set_correctly(self):
        self.assertEqual(self.rsvp._preview_inquiry, self.preview_inquiry)
        # In our test case, preview_inquiry_progress_method is 1 and should map to press_to_accept
        self.assertEqual(self.rsvp._preview_inquiry.press_to_accept, True)
        self.assertEqual(
            self.rsvp._preview_inquiry.preview_inquiry_length,
            self.preview_inquiry.preview_inquiry_length)
        self.assertEqual(
            self.rsvp._preview_inquiry.preview_inquiry_key_input,
            self.preview_inquiry.preview_inquiry_key_input)
        self.assertEqual(
            self.rsvp._preview_inquiry.preview_inquiry_isi,
            self.preview_inquiry.preview_inquiry_isi)

    def test_preview_inquiry_generate_inquiry_preview_fixation(self):
        text = ' '.join(self.rsvp.stimuli_inquiry).split('+ ')[1]
        stim_mock = mock()
        when(self.rsvp)._create_stimulus(
            any(),
            stimulus=text,
            units='height',
            stimuli_position=self.rsvp.stimuli_pos,
            mode='textbox',
            wrap_width=any()
        ).thenReturn(stim_mock)

        response = self.rsvp._generate_inquiry_preview()
        self.assertEqual(response, stim_mock)

    @patch('bcipy.display.paradigm.rsvp.display.get_key_press')
    def test_preview_inquiry_evoked_press_to_accept_pressed(self, get_key_press_mock):
        stim_mock = mock()
        # mock the stimulus generation
        when(self.rsvp)._generate_inquiry_preview().thenReturn(stim_mock)
        when(stim_mock).draw().thenReturn()
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(self.rsvp)._trigger_pulse().thenReturn()

        # skip the core wait for testing
        when(psychopy.core).wait(self.preview_inquiry.preview_inquiry_isi).thenReturn()
        key_timestamp = 1000
        get_key_press_mock.return_value = [
            f'bcipy_key_press_{self.preview_inquiry_key_input}', key_timestamp
        ]
        response = self.rsvp.preview_inquiry()
        # we expect the trigger callback to return none, the key press to return the time and key press message,
        #  The second item should be True as it is press to accept and a response was returned
        expected = ([None, [f'bcipy_key_press_{self.preview_inquiry_key_input}', key_timestamp]], True)
        self.assertEqual(response, expected)

    @patch('bcipy.display.paradigm.rsvp.display.get_key_press')
    def test_preview_inquiry_evoked_press_to_skip_pressed(self, get_key_press_mock):
        # set the progress method to press to skip
        self.rsvp._preview_inquiry.press_to_accept = False
        stim_mock = mock()
        # mock the stimulus generation
        when(self.rsvp)._generate_inquiry_preview().thenReturn(stim_mock)
        when(stim_mock).draw().thenReturn()
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(self.rsvp)._trigger_pulse().thenReturn()

        # skip the core wait for testing
        when(psychopy.core).wait(self.preview_inquiry.preview_inquiry_isi).thenReturn()
        key_timestamp = 1000
        get_key_press_mock.return_value = [
            f'bcipy_key_press_{self.preview_inquiry_key_input}', key_timestamp
        ]
        response = self.rsvp.preview_inquiry()
        # we expect the trigger callback to return none, the key press to return the time and key press message,
        #  The second item should be False as it is press to skip and a response was returned
        expected = ([None, [f'bcipy_key_press_{self.preview_inquiry_key_input}', key_timestamp]], False)
        self.assertEqual(response, expected)

    @patch('bcipy.display.paradigm.rsvp.display.get_key_press')
    def test_preview_inquiry_evoked_press_to_accept_not_pressed(self, get_key_press_mock):
        stim_mock = mock()
        # mock the stimulus generation
        when(self.rsvp)._generate_inquiry_preview().thenReturn(stim_mock)
        when(stim_mock).draw().thenReturn()
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(self.rsvp)._trigger_pulse().thenReturn()

        # skip the core wait for testing
        when(psychopy.core).wait(self.preview_inquiry.preview_inquiry_isi).thenReturn()
        get_key_press_mock.return_value = None
        response = self.rsvp.preview_inquiry()
        # we expect the trigger callback to return none, the key press to return the time and key press message,
        #  The second item should be False as it is press to accept and a response was returned
        expected = ([None], False)
        self.assertEqual(response, expected)

    @patch('bcipy.display.paradigm.rsvp.display.get_key_press')
    def test_preview_inquiry_evoked_press_to_skip_not_pressed(self, get_key_press_mock):
        # set the progress method to press to skip
        self.rsvp._preview_inquiry.press_to_accept = False
        stim_mock = mock()
        # mock the stimulus generation
        when(self.rsvp)._generate_inquiry_preview().thenReturn(stim_mock)
        when(stim_mock).draw().thenReturn()
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(self.rsvp)._trigger_pulse().thenReturn()

        # skip the core wait for testing
        when(psychopy.core).wait(self.preview_inquiry.preview_inquiry_isi).thenReturn()
        get_key_press_mock.return_value = None
        response = self.rsvp.preview_inquiry()
        # we expect the trigger callback to return none, the key press to return the time and key press message,
        #  The second item should be True as it is press to skip and a response was not returned
        expected = ([None], True)
        self.assertEqual(response, expected)

    @patch('bcipy.display.paradigm.rsvp.display.get_key_press')
    def test_preview_inquiry_preview_only_response_registered(self, get_key_press_mock):
        # set the progress method to press to skip
        self.rsvp._preview_inquiry.press_to_accept = False
        self.rsvp._preview_inquiry.preview_only = True
        stim_mock = mock()
        # mock the stimulus generation
        when(self.rsvp)._generate_inquiry_preview().thenReturn(stim_mock)
        when(stim_mock).draw().thenReturn()
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(self.rsvp)._trigger_pulse().thenReturn()

        # skip the core wait for testing
        when(psychopy.core).wait(self.preview_inquiry.preview_inquiry_isi).thenReturn()
        # we return a key press value here to demonstrate, even if a response is returned by this method, it will not
        #  be used in the preview_inquiry response from our display.
        key_timestamp = 1000
        get_key_press_mock.return_value = [
            f'bcipy_key_press_{self.preview_inquiry_key_input}', key_timestamp
        ]
        response = self.rsvp.preview_inquiry()
        # we expect the trigger callback to return none, no key press response even if returned,
        #  The second item should be True as it is preview only
        expected = ([None], True)
        self.assertEqual(response, expected)

    @patch('bcipy.display.paradigm.rsvp.display.get_key_press')
    def test_preview_inquiry_preview_only_no_response(self, get_key_press_mock):
        # set the progress method to press to skip
        self.rsvp._preview_inquiry.press_to_accept = False
        self.rsvp._preview_inquiry.preview_only = True
        stim_mock = mock()
        # mock the stimulus generation
        when(self.rsvp)._generate_inquiry_preview().thenReturn(stim_mock)
        when(stim_mock).draw().thenReturn()
        when(self.rsvp).draw_static().thenReturn()
        when(self.rsvp.window).flip().thenReturn()
        when(self.rsvp)._trigger_pulse().thenReturn()

        # skip the core wait for testing
        when(psychopy.core).wait(self.preview_inquiry.preview_inquiry_isi).thenReturn()

        get_key_press_mock.return_value = None
        response = self.rsvp.preview_inquiry()
        # we expect the trigger callback to return none, no key press response,
        #  The second item should be True as it is preview only
        expected = ([None], True)
        self.assertEqual(response, expected)

    def test_error_thrown_when_calling_preview_inquiry_without_properties_set(self):
        # If not defined using the kwarg preview_inquiry, this value is set to None
        self.rsvp._preview_inquiry = None

        # Assert when set to None, calling the method will result in an exception
        with self.assertRaises(Exception):
            self.rsvp.preview_inquiry()


if __name__ == '__main__':
    unittest.main()
