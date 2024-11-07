import logging
import os.path as path
from typing import List, Optional, Tuple

from psychopy import core, visual

from bcipy.display import (BCIPY_LOGO_PATH, Display, InformationProperties,
                           StimuliProperties)
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.main import PreviewParams, init_preview_button_handler
from bcipy.helpers.clock import Clock
from bcipy.data.stimuli import resize_image
from bcipy.data.symbols import SPACE_CHAR
from bcipy.helpers.utils import get_screen_info
from bcipy.data.triggers import TriggerCallback, _calibration_trigger


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
            preview_config: Optional[PreviewParams] = None,
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
        preview_config(PreviewParams) Optional: parameters used to specify the behavior for displaying a preview
            of upcoming stimuli defined via self.stimuli(StimuliProperties). If None a preview is not displayed.

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
        # Note: there is a bug in TextBox2 that prevents certain custom fonts from being used. This is to avoid that.
        self.textbox_font = 'Consolas'
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.is_txt_stim = stimuli.is_txt_stim
        self.stim_length = stimuli.stim_length

        self.full_screen = full_screen

        self.preview_params = preview_config
        self.preview_button_handler = init_preview_button_handler(
            preview_config, experiment_clock) if self.preview_enabled else None
        self.preview_accepted = True

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

    @property
    def preview_enabled(self) -> bool:
        """Should the inquiry preview be enabled."""
        return self.preview_params and self.preview_params.show_preview_inquiry

    def draw_static(self):
        """Draw static elements in a stimulus."""
        if self.task_bar:
            self.task_bar.draw()
        for info in self.info_text:
            info.draw()

    def schedule_to(self,
                    stimuli: Optional[List[str]] = None,
                    timing: Optional[List[float]] = None,
                    colors: Optional[List[str]] = None):
        """Schedule stimuli elements (works as a buffer).

        Args:
                stimuli(list[string]): list of stimuli text / name
                timing(list[float]): list of timings of stimuli
                colors(list[string]): list of colors
        """
        self.stimuli_inquiry = stimuli or []
        self.stimuli_timing = timing or []
        self.stimuli_colors = colors or []

    @property
    def preview_index(self) -> int:
        """Index within an inquiry at which the inquiry preview should be displayed.

        For calibration, we should display it after the target prompt (index = 1).
        For copy phrase there is no target prompt so it should display before the
        rest of the inquiry."""
        return 1

    def do_inquiry(self) -> List[Tuple[str, float]]:
        """Do inquiry.

        Animates an inquiry of flashing letters to achieve RSVP.


        RETURNS:
        --------
        timing(list[float]): list of timings of stimuli presented in the inquiry
        """

        # init an array for timing information
        timing: List[Tuple[str, float]] = []
        self.preview_accepted = True

        if self.first_run:
            self._trigger_pulse()

        # generate a inquiry (list of stimuli with meta information)
        inquiry = self._generate_inquiry()

        # do the inquiry
        for idx, stim_props in enumerate(inquiry):

            # If this is the start of an inquiry and a callback registered for first_stim_callback evoke it
            if idx == 0 and callable(self.first_stim_callback):
                self.first_stim_callback(stim_props['sti'])

            # If previewing the inquiry during calibration, do so after the first stimulus
            if self.preview_enabled and idx == self.preview_index:
                self.preview_accepted = self.preview_inquiry(timing)
            if not self.preview_accepted:
                break

            # Reset the timing clock to start presenting
            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                stim_props['sti_label'])

            # Draw stimulus for n frames
            stim_props['sti'].draw()
            self.draw_static()
            self.window.flip()
            core.wait(stim_props['time_to_present'])

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

    def preview_inquiry(self, timing: List[Tuple[str, float]]) -> bool:
        """Preview Inquiry.

        Given an inquiry defined to be presented via do_inquiry(), present the full inquiry
            to the user and allow input on whether the intended letter is present or not before
            going through the rapid serial visual presention.

        Parameters:
            timing - list to which all timing information should be appended.
        Returns:
            - A boolean describing whether to present the inquiry (True) or
                generate another (False).
        """
        assert self.preview_enabled, "Preview feature not enabled."
        assert self.preview_button_handler, "Button handler must be initialized"

        handler = self.preview_button_handler
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'inquiry_preview')
        self.draw_preview()

        handler.await_response()
        timing.append(self.trigger_callback.timing)
        if handler.has_response():
            timing.append((handler.response_label, handler.response_timestamp))

        self.trigger_callback.reset()
        self.draw_static()
        self.window.flip()
        core.wait(self.preview_params.preview_inquiry_isi)

        return handler.accept_result()

    def draw_preview(self):
        """Generate and draw the inquiry preview"""
        content = self._generate_inquiry_preview()
        content.draw()
        self.draw_static()
        self.window.flip()

    def _generate_inquiry_preview(self) -> visual.TextBox2:
        """Generate Inquiry Preview.

        Using the self.stimuli_inquiry list, construct a preview box to display to the user. This method
            assumes the presence of a fixation (+).
        """
        text = ' '.join(self.stimuli_inquiry).split('+ ')[1]

        return self._create_stimulus(
            self.preview_params.preview_box_text_size,
            stimulus=text,
            units='height',
            stimuli_position=self.stimuli_pos,
            mode='textbox',
            align_text='left')

    def _generate_inquiry(self) -> list:
        """Generate inquiry.

        Generate stimuli for next RSVP inquiry. [A + A, C, Q, D]
        """
        stim_info = []
        for idx, stim in enumerate(self.stimuli_inquiry):
            current_stim = {}

            current_stim['time_to_present'] = self.stimuli_timing[idx]

            # check if stimulus needs to use a non-default size
            if self.size_list_sti:
                this_stimuli_size = self.size_list_sti[idx]
            else:
                this_stimuli_size = self.stimuli_height

            # Set the Stimuli attrs
            if stim.endswith('.png'):
                current_stim['sti'] = self._create_stimulus(
                    mode='image',
                    height=this_stimuli_size,
                    stimulus=stim
                )
                current_stim['sti'].size = resize_image(
                    current_stim['sti'].image, current_stim['sti'].win.size, this_stimuli_size)
                current_stim['sti_label'] = path.splitext(
                    path.basename(stim))[0]
            else:
                # text stimulus
                current_stim['sti'] = self._create_stimulus(mode='text', height=this_stimuli_size)
                txt = stim
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

    def update_task_bar(self, text: Optional[str] = None) -> None:
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
                padding=0.05,
                font=self.textbox_font,
                letterHeight=height,
                size=[.55, .55],
                pos=stimuli_position,
                alignment=align_text,
                editable=False,
            )
