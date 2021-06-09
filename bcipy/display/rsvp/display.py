import logging
import os.path as path
from typing import List, Optional, Tuple, Union

from psychopy import core, visual

from bcipy.acquisition.marker_writer import NullMarkerWriter, MarkerWriter
from bcipy.helpers.task import SPACE_CHAR, get_key_press
from bcipy.display import Display
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.system_utils import get_screen_resolution
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger


class StimuliProperties:
    """"Stimuli Properties.

    An encapsulation of properties relevant to core stimuli presentation in an RSVP paradigm.
    """

    def __init__(
            self,
            stim_font: str,
            stim_pos: Tuple[float, float],
            stim_height: float,
            stim_inquiry: List[str],
            stim_colors: List[str],
            stim_timing: List[float],
            is_txt_stim: bool):
        """Initialize Stimuli Parameters.

        stim_font(List[str]): Ordered list of colors to apply to information stimuli
        stim_pos(Tuple[float, float]): Position on window where the stimuli will be presented
        stim_height(float): Height of all stimuli
        stim_inquiry(List[str]): Ordered list of text to build stimuli with
        stim_colors(List[str]): Ordered list of colors to apply to stimuli
        stim_timing(List[float]): Ordered list of timing to apply to an inquiry using the stimuli
        is_txt_stim(bool): Whether or not this is a text based stimuli (False implies image based)
        """
        self.stim_font = stim_font
        self.stim_pos = stim_pos
        self.stim_height = stim_height
        self.stim_inquiry = stim_inquiry
        self.stim_colors = stim_colors
        self.stim_timing = stim_timing
        self.is_txt_stim = is_txt_stim
        self.stim_length = len(self.stim_inquiry)
        self.sti = None

    def build_init_stimuli(self, window: visual.Window) -> Union[visual.TextStim, visual.ImageStim]:
        """"Build Initial Stimuli.

        This method constructs the stimuli object which can be updated later. This is more
            performant than creating a new stimuli each call. It can create either an image or text stimuli
            based on the boolean self.is_txt_stim.
        """
        if self.is_txt_stim:
            self.sti = visual.TextStim(
                win=window,
                color='white',
                height=self.stim_height,
                text='',
                font=self.stim_font,
                pos=self.stim_pos,
                wrapWidth=None, colorSpace='rgb',
                opacity=1, depth=-6.0)
        else:
            self.sti = visual.ImageStim(
                win=window,
                image=None,
                mask=None,
                pos=self.stim_pos,
                ori=0.0)
        return self.sti


class InformationProperties:
    """"Information Properties.

    An encapsulation of properties relevant to task information presentation in an RSVP paradigm. This could be
        messaging relevant to feedback or static text to remain on screen not related to task tracking.
    """

    def __init__(
            self,
            info_color: List[str],
            info_text: List[str],
            info_font: List[str],
            info_pos: Tuple[float, float],
            info_height: List[float]):
        """Initialize Information Parameters.

        info_color(List[str]): Ordered list of colors to apply to information stimuli
        info_text(List[str]): Ordered list of text to apply to information stimuli
        info_font(List[str]): Ordered list of font to apply to information stimuli
        info_pos(Tuple[float, float]): Position on window where the Information stimuli will be presented
        info_height(List[float]): Ordered list of height of Information stimuli
        """
        self.info_color = info_color
        self.info_text = info_text
        self.info_font = info_font
        self.info_pos = info_pos
        self.info_height = info_height

    def build_info_text(self, window: visual.Window) -> List[visual.TextStim]:
        """"Build Information Text.

        Constructs a list of Information stimuli to display.
        """
        self.text_stim = []
        for idx in range(len(self.info_text)):
            self.text_stim.append(visual.TextStim(
                win=window,
                color=self.info_color[idx],
                height=self.info_height[idx],
                text=self.info_text[idx],
                font=self.info_font[idx],
                pos=self.info_pos[idx],
                wrapWidth=None, colorSpace='rgb',
                opacity=1, depth=-6.0))
        return self.text_stim


class TaskDisplayProperties:
    """"Task Dispay Properties.

    An encapsulation of properties relevant to task stimuli presentation in an RSVP paradigm.
    """

    def __init__(
            self,
            task_color: List[str],
            task_font: str,
            task_pos: Tuple[float, float],
            task_height: float,
            task_text: str):
        """Initialize Task Display Parameters.

        task_color(List[str]): Ordered list of colors to apply to task stimuli
        task_font(str): Font to apply to all task stimuli
        task_pos(Tuple[float, float]): Position on the screen where to present to task text
        task_height(float): Height of all task text stimuli
        task_text(str): Task text to apply to stimuli
        """
        self.task_color = task_color
        self.task_font = task_font
        self.task_pos = task_pos
        self.task_height = task_height
        self.task_text = task_text
        self.task = None

    def build_task(self, window: visual.Window) -> visual.TextStim:
        """"Build Task.

        This method constructs the task stimuli object which can be updated later. This is more
            performant than creating a new stimuli for each update in task state.
        """
        self.task = visual.TextStim(
            win=window,
            color=self.task_color[0],
            height=self.task_height,
            text=self.task_text,
            font=self.task_font,
            pos=self.task_pos,
            wrapWidth=None, colorSpace='rgb',
            opacity=1, depth=-6.0)
        return self.task


class PreviewInquiryProperties:
    """"Preview Inquiry Properties.

    An encapsulation of properties relevant to preview_inquiry() operation.
    """

    def __init__(
            self,
            preview_inquiry_length: float,
            preview_inquiry_progress_method: int,
            preview_inquiry_key_input: str,
            preview_inquiry_isi: float):
        """Initialize Inquiry Preview Parameters.

        preview_inquiry_length(float): Length of time in seconds to present the inquiry preview
        preview_inquiry_progress_method(int): Method of progression for inquiry preview. 1 == press to accept
            inquiry 2 == press to skip inquiry
        preview_inquiry_key_input(str): Defines which key should be listened to for progressing
        preview_inquiry_isi(float): Length of time after displaying the inquiry preview to display a blank screen
        """
        self.preview_inquiry_length = preview_inquiry_length
        self.preview_inquiry_key_input = preview_inquiry_key_input
        self.press_to_accept = True if preview_inquiry_progress_method == 1 else False
        self.preview_inquiry_isi = preview_inquiry_isi


class RSVPDisplay(Display):
    """RSVP Display Object for inquiry Presentation.

    Animates display objects common to any RSVP task.
    """

    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: core.Clock,
            stimuli: StimuliProperties,
            task_display: TaskDisplayProperties,
            info: InformationProperties,
            preview_inquiry: PreviewInquiryProperties = None,
            marker_writer: Optional[MarkerWriter] = NullMarkerWriter(),
            trigger_type: str = 'image',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False):
        """Initialize RSVP display parameters and objects.

        PARAMETERS:
        ----------
        # Experiment
        window(visual.Window): PsychoPy Window
        static_clock(core.Clock): Used to schedule static periods of display time
        experiment_clock(core.Clock): Clock used to timestamp display onsets

        # Stimuli
        stimuli(StimuliProperties): attributes used for inquiries

        # Task
        task_display(TaskDisplayProperties): attributes used for task tracking. Ex. 1/100

        # Info
        info(InformationProperties): attributes to display informational stimuli alongside task and inquiry stimuli.

        # Preview Inquiry
        preview_inquiry(PreviewInquiryProperties) Optional: attributes to display a preview of upcoming stimuli defined
            via self.stimuli(StimuliProperties).

        marker_writer(MarkerWriter) Optional: object used to write triggers to
            a acquisition stream.
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
        #  easy updating after defintion
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
        self.marker_writer = marker_writer or NullMarkerWriter()
        self.experiment_clock = experiment_clock

        # Callback used on presentation of first stimulus.
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []  # TODO force initial size definition
        self.space_char = space_char  # TODO remove and force task to define
        self.task_display = task_display
        self.task = task_display.build_task(self.window)

        # Create multiple text objects based on input
        self.info = info
        self.text = info.build_info_text(window)

        # Create initial stimuli object for updating
        self.sti = stimuli.build_init_stimuli(window)

    def draw_static(self):
        """Draw static elements in a stimulus."""
        self.task.draw()
        for idx in range(len(self.text)):
            self.text[idx].draw()

    def schedule_to(self, stimuli=[], timing=[], colors=[]):
        """Schedule stimuli elements (works as a buffer).

        Args:
                stimuli(list[string]): list of stimuli text / name
                timing(list[float]): list of timings of stimuli
                colors(list[string]): list of colors
        """
        self.stimuli_inquiry = stimuli
        self.stimuli_timing = timing
        self.stimuli_colors = colors

    def update_task(self, text: str, color_list: List[str], pos: Tuple[float]):
        """Update Task Object.

        PARAMETERS:
        -----------
        text: text for task
        color_list: list of the colors for each char
        pos: position of task
        """
        self.task.text = text
        self.task.color = color_list[0]
        self.task.pos = pos

    def do_inquiry(self) -> List[float]:
        """Do inquiry.

        Animates an inquiry of flashing letters to achieve RSVP.
        """

        # init an array for timing information
        timing = []

        if self.first_run:
            timing = self._trigger_pulse(timing)

        # generate a inquiry (list of stimuli with meta information)
        inquiry = self._generate_inquiry()

        # do the inquiry
        for idx in range(len(inquiry)):

            # set a static period to do all our stim setting.
            #   will warn if ISI value is violated.
            self.staticPeriod.name = 'Stimulus Draw Period'
            self.staticPeriod.start(self.stimuli_timing[idx])

            # Reset the timing clock to start presenting
            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                inquiry[idx]['sti_label'])
            self.window.callOnFlip(self.marker_writer.push_marker, inquiry[idx]['sti_label'])

            # If this is the start of an inquiry and a callback registered for first_stim_callback evoke it
            if idx == 0 and callable(self.first_stim_callback):
                self.first_stim_callback(inquiry[idx]['sti'])

            # Draw stimulus for n frames
            inquiry[idx]['sti'].draw()
            self.draw_static()
            self.window.flip()
            core.wait(inquiry[idx]['time_to_present'])

            # End static period
            self.staticPeriod.complete()

            # append timing information
            if self.is_txt_stim:
                timing.append(self.trigger_callback.timing)
            else:
                timing.append(self.trigger_callback.timing)

            self.trigger_callback.reset()

        # draw in static and flip once more
        self.draw_static()
        self.window.flip()

        return timing

    def _trigger_pulse(self, timing: List[str]) -> List[str]:
        """Trigger Pulse.

        This method uses a marker writer and calibration trigger to determine any functional
            offsets needed for operation with this display. By setting the first_stim_time and searching for the
            same stimuli output to the marker stream, the offsets between these proceses can be reconciled at the
            beginning of an experiment. If drift is detected in your experiment, more frequent pulses and offset
            correction may be required.
        """
        calibration_time = _calibration_trigger(
            self.experiment_clock,
            trigger_type=self.trigger_type,
            display=self.window,
            on_trigger=self.marker_writer.push_marker)

        timing.append(calibration_time)

        # set the first stim time if not present and first_run to False
        if not self.first_stim_time:
            self.first_stim_time = calibration_time[-1]
            self.first_run = False

        return timing

    def preview_inquiry(self) -> Tuple[List[float], bool]:
        """Preview Inquiry.

        Given an inquiry defined to be presented via do_inquiry(), present the full inquiry
            to the user and allow input on whether the intended letter is present or not before
            going through the rapid serial visual presention.

        Returns:
            - A tuple containing the timing information and a boolean describing whether to present
                the inquiry (True) or generate another (False).
        """
        # construct the timing to return and generate the content for preview
        timing = []
        if self.first_run:
            timing = self._trigger_pulse(timing)

        content = self._generate_inquiry_preview()

        # define the trigger callbacks. Here we use the trigger_callback to return immediately and a marker_writer
        # which can link to external acquisition devices
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'inquiry_preview')
        self.window.callOnFlip(self.marker_writer.push_marker, 'inquiry_preview')

        # Draw and flip the screen.
        content.draw()
        self.draw_static()
        self.window.flip()
        timing.append(self.trigger_callback.timing)

        timer = core.CountdownTimer(self._preview_inquiry.preview_inquiry_length)
        response = False
        while timer.getTime() > 0:
            # wait for a key press event
            response = get_key_press(
                key_list=[self._preview_inquiry.preview_inquiry_key_input],
                clock=self.experiment_clock,
            )
            if response:
                break

        # reset the screen
        self.draw_static()
        self.window.flip()
        self.trigger_callback.reset()

        core.wait(self._preview_inquiry.preview_inquiry_isi)

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
        if not self.full_screen:
            reduce_factor = 4.85
            wrap_width = 1.1
        else:
            reduce_factor = 4.75
            wrap_width = .9
        text = ' '.join(self.stimuli_inquiry).split('+ ')[1]

        return self._create_stimulus(
            self.stimuli_height / reduce_factor,
            stimulus=text,
            units='height',
            stimuli_position=self.stimuli_pos,
            mode='textbox',
            wrap_width=wrap_width)

    def _generate_inquiry(self):
        """Generate inquiry.

        Generate stimuli for next RSVP inquiry.
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
                current_stim['sti'] = self._create_stimulus(mode='image', height=this_stimuli_size)
                current_stim['sti'].image = self.stimuli_inquiry[idx]
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
                    monitor_width, monitor_height = get_screen_resolution()
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

    def update_task_state(self, text: str, color_list: List[str]) -> None:
        """Update task state.

        Removes letters or appends to the right.
        Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each
        """
        task_state_text = visual.TextStim(
            win=self.window, font=self.task.font, text=text)
        x_task_position = task_state_text.boundingBox[0] / \
            self.window.size[0] - 1
        task_pos = (x_task_position, 1 - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=task_pos)

    def wait_screen(self, message, color):
        """Wait Screen.

        Args:
            message(string): message to be displayed while waiting
        """

        # Construct the wait message
        wait_message = visual.TextStim(win=self.window, font=self.stimuli_font,
                                       text=message,
                                       height=.1,
                                       color=color,
                                       pos=(0, -.5),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1, depth=-6.0)

        # Try adding our BCI logo. Pass if not found.
        try:
            wait_logo = visual.ImageStim(
                self.window,
                image='bcipy/static/images/gui/bci_cas_logo.png',
                pos=(0, .5),
                mask=None,
                ori=0.0)
            wait_logo.size = resize_image(
                'bcipy/static/images/gui/bci_cas_logo.png',
                self.window.size, 1)
            wait_logo.draw()

        except Exception:
            self.logger.debug('Cannot load logo image')
            pass

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

        Returns a TextStim or ImageStim object.
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
                anchor=align_text,
                alignment=align_text,
            )
