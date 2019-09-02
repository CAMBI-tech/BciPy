import logging
import os.path as path
from typing import List, Optional, Tuple

from psychopy import core, visual

from bcipy.acquisition.marker_writer import NullMarkerWriter, MarkerWriter
from bcipy.helpers.task import SPACE_CHAR
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.system_utils import get_system_info
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger


class RSVPDisplay(object):
    """RSVP Display Object for Sequence Presentation.

    Animates a sequence in RSVP. Mode should be determined outside.
    """

    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: core.Clock,
            marker_writer: Optional[MarkerWriter] = None,
            task_color: List[str] = ['white'],
            task_font: str = 'Times',
            task_pos: Tuple[float, float] = (-.8, .9),
            task_height: float = 0.2,
            task_text: str = '1/100',
            info_color: List[str] = ['white'],
            info_text: List[str] = ['Information Text'],
            info_font: List[str] = ['Times'],
            info_pos=[(.8, .9)],
            info_height=[0.2],
            stim_font='Times',
            stim_pos=(-.8, .9),
            stim_height=0.2,
            stim_sequence: List[str] = ['a'] * 10,
            stim_colors: List[str] = ['white'] * 10,
            stim_timing: List[float] = [1] * 10,
            is_txt_stim: bool = True,
            static_time: float = .05,
            trigger_type: str = 'image',
            space_char: SPACE_CHAR = SPACE_CHAR):
        """Initialize RSVP window parameters and objects.

        PARAMETERS:
        ----------
        # Experiment
        window(visual.Window): PsychoPy Window
        static_clock(TODO): no idea
        experiment_clock(core.Clock): Clock used to timestamp experiment
        marker_writer(MarkerWriter): object used to write triggers to
            the daq stream.

        # Task
        task_color(list[string]): Color of the task string. Shares the
            length of the task_text. If of length 1 the entire task
            bar shares the same color.
        task_font(string): Font of task string
        task_pos(tuple): position of task string
        task_height(float): height for task string
        task_text(string): text of the task bar

        # Info
        info_text(list[string]): Text list for information texts
        info_color(list[string]): Color of the information text string
        info_font(list[string]): Font of the information text string
        info_pos(list[tuple]): Position of the information text string
        info_height(list[float]): Height of the information text string

        # Stimuli
        stim_height(float): height of the stimuli object
        stim_pos(tuple): position of stimuli
        stim_font(string): font of the stimuli
        stim_sequence(list[string]): list of elements to flash
        stim_colors(list[string]): list of colors for stimuli
        stim_timing(list[float]): timing for each letter flash
        """
        self.window = window
        self.refresh_rate = window.getActualFrameRate()

        self.logger = logging.getLogger(__name__)

        self.stimuli_sequence = stim_sequence
        self.stimuli_colors = stim_colors
        self.stimuli_timing = stim_timing

        self.is_txt_stim = is_txt_stim
        self.staticPeriod = static_clock
        self.static_time = static_time
        self.experiment_clock = experiment_clock
        self.timing_clock = core.Clock()

        # Used to handle writing the marker stimulus
        self.marker_writer = marker_writer or NullMarkerWriter()

        # Length of the stimuli (number of flashes)
        self.stim_length = len(stim_sequence)

        # Informational Parameters
        self.info_text = info_text

        # Stim parameters
        self.stimuli_font = stim_font
        self.stimuli_height = stim_height
        self.stimuli_pos = stim_pos

        # Trigger Items
        self.first_run = True
        self.trigger_type = trigger_type
        self.trigger_callback = TriggerCallback()

        # Callback used on presentation of first stimulus.
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []

        self.space_char = space_char

        self.task = visual.TextStim(win=self.window, color=task_color[0],
                                    height=task_height,
                                    text=task_text,
                                    font=task_font, pos=task_pos,
                                    wrapWidth=None, colorSpace='rgb',
                                    opacity=1, depth=-6.0)

        # Create multiple text objects based on input
        self.text = []
        for idx in range(len(self.info_text)):
            self.text.append(visual.TextStim(
                win=self.window,
                color=info_color[idx],
                height=info_height[idx],
                text=self.info_text[idx],
                font=info_font[idx],
                pos=info_pos[idx],
                wrapWidth=None, colorSpace='rgb',
                opacity=1, depth=-6.0))

        # Create Stimuli Object
        if self.is_txt_stim:
            self.sti = visual.TextStim(
                win=self.window,
                color='white',
                height=self.stimuli_height,
                text='+',
                font=self.stimuli_font,
                pos=self.stimuli_pos,
                wrapWidth=None, colorSpace='rgb',
                opacity=1, depth=-6.0)
        else:
            self.sti = visual.ImageStim(
                win=self.window,
                image=None,
                mask=None,
                pos=self.stimuli_pos,
                ori=0.0)

    def draw_static(self):
        """Draw static elements in a stimulus."""
        self.task.draw()
        for idx in range(len(self.text)):
            self.text[idx].draw()

    def schedule_to(self, ele_list=[], time_list=[], color_list=[]):
        """Schedule stimuli elements (works as a buffer).

        Args:
                ele_list(list[string]): list of elements of stimuli
                time_list(list[float]): list of timings of stimuli
                color_list(list[string]): colors of elements of stimuli
        """
        self.stimuli_sequence = ele_list
        self.stimuli_timing = time_list
        self.stimuli_colors = color_list

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

    def do_sequence(self):
        """Do Sequence.

        Animates a sequence of flashing letters to achieve RSVP.
        """

        # init an array for timing information
        timing = []

        if self.first_run:
            # play a sequence start sound to help orient triggers
            first_stim_timing = _calibration_trigger(
                self.experiment_clock,
                trigger_type=self.trigger_type, display=self.window,
                on_trigger=self.marker_writer.push_marker)

            timing.append(first_stim_timing)

            self.first_stim_time = first_stim_timing[-1]
            self.first_run = False

        # generate a sequence (list of stimuli with meta information)
        sequence = self._generate_sequence()

        # do the sequence
        for idx in range(len(sequence)):

            self.is_first_stim = (idx == 0)

            # set a static period to do all our stim setting.
            #   will warn if ISI value is violated.
            self.staticPeriod.name = 'Stimulus Draw Period'
            self.staticPeriod.start(self.stimuli_timing[idx])

            # Reset the timing clock to start presenting
            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                sequence[idx]['sti_label'])
            self.window.callOnFlip(self.marker_writer.push_marker, sequence[idx]['sti_label'])

            if idx == 0 and callable(self.first_stim_callback):
                self.first_stim_callback(sequence[idx]['sti'])

            # Draw stimulus for n frames
            sequence[idx]['sti'].draw()
            self.draw_static()
            self.window.flip()
            core.wait((sequence[idx]['time_to_present'] - 1) / self.refresh_rate)

            # End static period
            # print(self.staticPeriod.countdown.getTime())
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

    def _generate_sequence(self):
        """Generate Sequence.

        Generate stimuli for next RSVP sequence.
        """
        stim_info = []
        for idx in range(len(self.stimuli_sequence)):
            current_stim = {}

            # turn ms timing into frames! Much more accurate!
            current_stim['time_to_present'] = int(self.stimuli_timing[idx] * self.refresh_rate)

            # check if stimulus needs to use a non-default size
            if self.size_list_sti:
                this_stimuli_size = self.size_list_sti[idx]
            else:
                this_stimuli_size = self.stimuli_height

            # Set the Stimuli attrs
            if self.stimuli_sequence[idx].endswith('.png'):
                current_stim['sti'] = self.create_stimulus(mode='image', height_int=this_stimuli_size)
                current_stim['sti'].image = self.stimuli_sequence[idx]
                current_stim['sti'].size = resize_image(
                    current_stim['sti'].image, current_stim['sti'].win.size, this_stimuli_size)
                current_stim['sti_label'] = path.splitext(
                    path.basename(self.stimuli_sequence[idx]))[0]
            else:
                # text stimulus
                current_stim['sti'] = self.create_stimulus(mode='text', height_int=this_stimuli_size)
                txt = self.stimuli_sequence[idx]
                # customize presentation of space char.
                current_stim['sti'].text = txt if txt != SPACE_CHAR else self.space_char
                current_stim['sti'].color = self.stimuli_colors[idx]
                current_stim['sti_label'] = txt

                # test whether the word will be too big for the screen
                text_width = current_stim['sti'].boundingBox[0]
                if text_width > self.window.size[0]:
                    info = get_system_info()
                    text_height = current_stim['sti'].boundingBox[1]
                    # If we are in full-screen, text size in Psychopy norm units
                    # is monitor width/monitor height
                    if self.window.size[0] == info['RESOLUTION'][0]:
                        new_text_width = info['RESOLUTION'][0] / \
                            info['RESOLUTION'][1]
                    else:
                        # If not, text width is calculated relative to both
                        # monitor size and window size
                        new_text_width = (
                            self.window.size[1] / info['RESOLUTION'][1]) * (
                                info['RESOLUTION'][0] / info['RESOLUTION'][1])
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
                image='bcipy/static/images/gui_images/bci_cas_logo.png',
                pos=(0, .5),
                mask=None,
                ori=0.0)
            wait_logo.size = resize_image(
                'bcipy/static/images/gui_images/bci_cas_logo.png',
                self.window.size, 1)
            wait_logo.draw()

        except Exception:
            self.logger.debug('Cannot load logo image')
            pass

        # Draw and flip the screen.
        wait_message.draw()
        self.window.flip()

    def create_stimulus(self, height_int: int, mode: str = 'text'):
        """Create Stimulus.

        Returns a TextStim or ImageStim object.
            Args:
            height_int: The height of the stimulus
            mode: "text" or "image", determines which to return
        """
        if mode == 'text':
            return visual.TextStim(
                win=self.window,
                color='white',
                height=height_int,
                text='+',
                font=self.stimuli_font,
                pos=self.stimuli_pos,
                wrapWidth=None,
                colorSpace='rgb',
                opacity=1,
                depth=-6.0)
        if mode == 'image':
            return visual.ImageStim(
                win=self.window,
                image=None,
                mask=None,
                units='',
                pos=self.stimuli_pos,
                size=(height_int, height_int),
                ori=0.0)
