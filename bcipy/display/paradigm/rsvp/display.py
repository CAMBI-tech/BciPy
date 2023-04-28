import logging
import os.path as path
from typing import List, Tuple

from psychopy import core, visual, event

from bcipy.helpers.clock import Clock
from bcipy.helpers.task import get_key_press
from bcipy.helpers.symbols import SPACE_CHAR
from bcipy.display import (
    BCIPY_LOGO_PATH,
    Display,
    InformationProperties,
    PreviewInquiryProperties,
    StimuliProperties
)
from bcipy.display.components.task_bar import TaskBar
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.system_utils import get_screen_info
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger


class RSVPDisplay(Display):
    """RSVP Display Object for inquiry Presentation.

    Animates display objects common to any RSVP task.
    """

    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: Clock,
            stimuli: StimuliProperties,
            task_bar: TaskBar,
            info: InformationProperties,
            preview_inquiry: PreviewInquiryProperties = None,
            trigger_type: str = 'image',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False):
        """Initialize RSVP display parameters and objects.

        PARAMETERS:
        ----------
        # Experiment
        window(visual.Window): PsychoPy Window
        static_clock(core.MonotonicClock): Used to schedule static periods of display time
        experiment_clock(Clock): Clock used to timestamp display onsets

        # Stimuli
        stimuli(StimuliProperties): attributes used for inquiries

        # Task
        task_bar(TaskBar): used for task tracking. Ex. 1/100

        # Info
        info(InformationProperties): attributes to display informational stimuli alongside task and inquiry stimuli.

        # Preview Inquiry
        preview_inquiry(PreviewInquiryProperties) Optional: attributes to display a preview of upcoming stimuli defined
            via self.stimuli(StimuliProperties).

        trigger_type(str) default 'image': defines the calibration trigger type for the display at the beginning of any
            task. This will be used to reconcile timing differences between acquisition and the display.
        space_char(str) default SPACE_CHAR: defines the space character to use in the RSVP inquiry.
        full_screen(bool) default False: Whether or not the window is set to a full screen dimension. Used for
            scaling display items as needed.
        """
        self.window = window
        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = window.getActualFrameRate()

        self.logger = logging.getLogger(__name__)

        # Stimuli parameters, these are set on display in order to allow
        #  easy updating after definition
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.stimuli_font = stimuli.stim_font
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.is_txt_stim = stimuli.is_txt_stim
        self.stim_length = stimuli.stim_length

        self.full_screen = full_screen
        self._preview_inquiry = preview_inquiry

        self.staticPeriod = static_clock

        # Trigger handling
        self.first_run = True
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self.trigger_callback = TriggerCallback()
        self.experiment_clock = experiment_clock

        # Callback used on presentation of first stimulus.
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []  # TODO force initial size definition
        self.space_char = space_char  # TODO remove and force task to define

        self.task_bar = task_bar

        # Create multiple text objects based on input
        self.info = info
        self.info_text = info.build_info_text(window)

        # Create initial stimuli object for updating
        self.sti = stimuli.build_init_stimuli(window)

    def draw_static(self):
        """Draw static elements in a stimulus."""
        if self.task_bar:
            self.task_bar.draw()
        for info in self.info_text:
            info.draw()

    def schedule_to(self,
                    stimuli: List[str] = None,
                    timing: List[float] = None,
                    colors: List[str] = None):
        """Schedule stimuli elements (works as a buffer).

        Args:
                stimuli(list[string]): list of stimuli text / name
                timing(list[float]): list of timings of stimuli
                colors(list[string]): list of colors
        """
        self.stimuli_inquiry = stimuli or []
        self.stimuli_timing = timing or []
        self.stimuli_colors = colors or []

    def do_inquiry(self, preview_calibration: bool = False) -> List[float]:
        """Do inquiry.

        Animates an inquiry of flashing letters to achieve RSVP.


        PARAMETERS:
        -----------
        preview_calibration(bool) default False: Whether or not to preview the upcoming inquiry stimuli. This feature
            is used to help the participant prepare for the upcoming inquiry after a prompt. It will present after
            the first stimulus of the inquiry (assumed to be a prompt). Not recommended for use outside of a
            calibration task.

        RETURNS:
        --------
        timing(list[float]): list of timings of stimuli presented in the inquiry
        """

        # init an array for timing information
        timing = []

        if self.first_run:
            self._trigger_pulse()

        # generate a inquiry (list of stimuli with meta information)
        inquiry = self._generate_inquiry()

        # do the inquiry
        for idx in range(len(inquiry)):

            # If this is the start of an inquiry and a callback registered for first_stim_callback evoke it
            if idx == 0 and callable(self.first_stim_callback):
                self.first_stim_callback(inquiry[idx]['sti'])

            # If previewing the inquiry, do so after the first stimulus
            if preview_calibration and idx == 1:
                time, _ = self.preview_inquiry()
                timing.extend(time)

            # Reset the timing clock to start presenting
            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                inquiry[idx]['sti_label'])

            # Draw stimulus for n frames
            inquiry[idx]['sti'].draw()
            self.draw_static()
            self.window.flip()
            core.wait(inquiry[idx]['time_to_present'])

            # append timing information
            timing.append(self.trigger_callback.timing)

            self.trigger_callback.reset()

        # draw in static and flip once more
        self.draw_static()
        self.window.flip()

        return timing

    def _trigger_pulse(self) -> None:
        """Trigger Pulse.

        This method uses a calibration trigger to determine any functional
            offsets needed for operation with this display. By setting the first_stim_time and searching for the
            same stimuli output to the marker stream, the offsets between these proceses can be reconciled at the
            beginning of an experiment. If drift is detected in your experiment, more frequent pulses and offset
            correction may be required.
        """
        calibration_time = _calibration_trigger(
            self.experiment_clock,
            trigger_type=self.trigger_type,
            display=self.window)

        # set the first stim time if not present and first_run to False
        if not self.first_stim_time:
            self.first_stim_time = calibration_time[-1]
            self.first_run = False

    def preview_inquiry(self) -> Tuple[List[float], bool]:
        """Preview Inquiry.

        Given an inquiry defined to be presented via do_inquiry(), present the full inquiry
            to the user and allow input on whether the intended letter is present or not before
            going through the rapid serial visual presention.

        Returns:
            - A tuple containing the timing information and a boolean describing whether to present
                the inquiry (True) or generate another (False).
        """
        # self._preview_inquiry defaults to None on __init__, assert it is defined correctly
        assert isinstance(self._preview_inquiry, PreviewInquiryProperties), (
            'PreviewInquiryProperties are not set on this RSVPDisplay. '
            'Add them as a preview_inquiry kwarg to use preview_inquiry().')
        # construct the timing to return and generate the content for preview
        timing = []
        if self.first_run:
            self._trigger_pulse()

        content = self._generate_inquiry_preview()

        # define the trigger callbacks. Here we use the trigger_callback to return immediately
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'inquiry_preview')

        # Draw and flip the screen.
        content.draw()
        self.draw_static()
        self.window.flip()
        timing.append(self.trigger_callback.timing)

        timer = core.CountdownTimer(self._preview_inquiry.preview_inquiry_length)
        response = False

        event.clearEvents(eventType='keyboard')
        while timer.getTime() > 0:
            # wait for a key press event
            response = get_key_press(
                key_list=[self._preview_inquiry.preview_inquiry_key_input],
                clock=self.experiment_clock,
            )

            # break if a response given unless this is preview only and wait the timer
            if response and not self._preview_inquiry.preview_only:
                break

        self.draw_static()
        self.window.flip()
        self.trigger_callback.reset()
        core.wait(self._preview_inquiry.preview_inquiry_isi)

        if self._preview_inquiry.preview_only:
            return timing, True

        # depending on whether or not press to accept, define what to return to the task
        if response and self._preview_inquiry.press_to_accept:
            timing.append(response)
            return timing, True
        elif response and not self._preview_inquiry.press_to_accept:
            timing.append(response)
            return timing, False
        elif not response and self._preview_inquiry.press_to_accept:
            return timing, False
        else:
            return timing, True

    def _generate_inquiry_preview(self) -> visual.TextBox2:
        """Generate Inquiry Preview.

        Using the self.stimuli_inquiry list, construct a preview box to display to the user. This method
            assumes the presence of a fixation (+).
        """
        text = ' '.join(self.stimuli_inquiry).split('+ ')[1]

        return self._create_stimulus(
            0.12,
            stimulus=text,
            units='height',
            stimuli_position=self.stimuli_pos,
            mode='textbox',
            align_text='left',
            wrap_width=0.025)

    def _generate_inquiry(self) -> list:
        """Generate inquiry.

        Generate stimuli for next RSVP inquiry. [A + A, C, Q, D]
        """
        stim_info = []
        for idx in range(len(self.stimuli_inquiry)):
            current_stim = {}

            current_stim['time_to_present'] = self.stimuli_timing[idx]

            # check if stimulus needs to use a non-default size
            if self.size_list_sti:
                this_stimuli_size = self.size_list_sti[idx]
            else:
                this_stimuli_size = self.stimuli_height

            # Set the Stimuli attrs
            if self.stimuli_inquiry[idx].endswith('.png'):
                current_stim['sti'] = self._create_stimulus(
                    mode='image',
                    height=this_stimuli_size,
                    stimulus=self.stimuli_inquiry[idx]
                )
                current_stim['sti'].size = resize_image(
                    current_stim['sti'].image, current_stim['sti'].win.size, this_stimuli_size)
                current_stim['sti_label'] = path.splitext(
                    path.basename(self.stimuli_inquiry[idx]))[0]
            else:
                # text stimulus
                current_stim['sti'] = self._create_stimulus(mode='text', height=this_stimuli_size)
                txt = self.stimuli_inquiry[idx]
                # customize presentation of space char.
                current_stim['sti'].text = txt if txt != SPACE_CHAR else self.space_char
                current_stim['sti'].color = self.stimuli_colors[idx]
                current_stim['sti_label'] = txt

                # test whether the word will be too big for the screen
                text_width = current_stim['sti'].boundingBox[0]
                if text_width > self.window.size[0]:
                    screen_info = get_screen_info()
                    monitor_width = screen_info.width
                    monitor_height = screen_info.height
                    text_height = current_stim['sti'].boundingBox[1]
                    # If we are in full-screen, text size in Psychopy norm units
                    # is monitor width/monitor height
                    if self.window.size[0] == monitor_width:
                        new_text_width = monitor_width / monitor_height
                    else:
                        # If not, text width is calculated relative to both
                        # monitor size and window size
                        new_text_width = (
                            self.window.size[1] / monitor_height) * (
                                monitor_width / monitor_height)
                    new_text_height = (text_height * new_text_width) / text_width
                    current_stim['sti'].height = new_text_height
            stim_info.append(current_stim)
        return stim_info

    def update_task_bar(self, text: str = None) -> None:
        """Update task state.

        Removes letters or appends to the right.
        Args:
                text(string): new text for task state
        """
        if self.task_bar:
            self.task_bar.update(text)

    def wait_screen(self, message: str, message_color: str) -> None:
        """Wait Screen.

        Args:
            message(string): message to be displayed while waiting
            message_color(string): color of the message to be displayed
        """

        # Construct the wait message
        wait_message = visual.TextStim(win=self.window,
                                       font=self.stimuli_font,
                                       text=message,
                                       height=.1,
                                       color=message_color,
                                       pos=(0, -.5),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1,
                                       depth=-6.0)

        # try adding the BciPy logo to the wait screen
        try:
            wait_logo = visual.ImageStim(
                self.window,
                image=BCIPY_LOGO_PATH,
                pos=(0, .25),
                mask=None,
                ori=0.0)
            wait_logo.size = resize_image(
                BCIPY_LOGO_PATH,
                self.window.size,
                1)
            wait_logo.draw()

        except Exception as e:
            self.logger.exception(f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            raise e

        # Draw and flip the screen.
        wait_message.draw()
        self.window.flip()

    def _create_stimulus(
            self,
            height: int,
            mode: str = 'text',
            stimulus='+',
            color='white',
            stimuli_position=None,
            align_text='center',
            units=None,
            wrap_width=None,
            border=False):
        """Create Stimulus.

        Returns a TextStim, ImageStim or TextBox object.
        """
        if not stimuli_position:
            stimuli_position = self.stimuli_pos
        if mode == 'text':
            return visual.TextStim(
                win=self.window,
                color=color,
                height=height,
                text=stimulus,
                font=self.stimuli_font,
                pos=stimuli_position,
                wrapWidth=wrap_width,
                colorSpace='rgb',
                units=units,
                alignText=align_text,
                opacity=1,
                depth=-6.0)
        elif mode == 'image':
            return visual.ImageStim(
                win=self.window,
                image=stimulus,
                mask=None,
                units=units,
                pos=stimuli_position,
                size=(height, height),
                ori=0.0)
        elif mode == 'textbox':
            return visual.TextBox2(
                win=self.window,
                text=stimulus,
                color=color,
                colorSpace='rgb',
                borderWidth=2,
                borderColor='white',
                units=units,
                font=self.stimuli_font,
                letterHeight=height,
                size=[.5, .5],
                pos=stimuli_position,
                alignment=align_text,
            )
