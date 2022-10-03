from typing import List, Optional, Tuple
import logging

from psychopy import visual, core

from bcipy.acquisition.marker_writer import NullMarkerWriter, MarkerWriter
from bcipy.display import Display, StimuliProperties, TaskDisplayProperties, InformationProperties, BCIPY_LOGO_PATH
from bcipy.helpers.task import SPACE_CHAR
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger
from bcipy.helpers.task import alphabet
from bcipy.helpers.exceptions import BciPyCoreException


class MatrixDisplay(Display):
    """Matrix Display Object for Inquiry Presentation.

    Animates display objects in matrix grid common to any Matrix task.
    """

    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: core.Clock,
            stimuli: StimuliProperties,
            task_display: TaskDisplayProperties,
            info: InformationProperties,
            marker_writer: Optional[MarkerWriter] = NullMarkerWriter(),
            trigger_type: str = 'text',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False,
            symbol_set: Optional[List[str]] = None):
        """Initialize Matrix display parameters and objects.

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

        marker_writer(MarkerWriter) Optional: object used to write triggers to an acquisition stream.
        trigger_type(str) default 'image': defines the calibration trigger type for the display at the beginning of any
            task. This will be used to reconcile timing differences between acquisition and the display.
        space_char(str) default SPACE_CHAR: defines the space character to use in the Matrix inquiry.
        full_screen(bool) default False: Whether or not the window is set to a full screen dimension. Used for
            scaling display items as needed.
        symbol_set default = none : subset of stimuli to be highlighted during an inquiry
        """
        self.window = window
        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = window.getActualFrameRate()

        self.logger = logging.getLogger(__name__)

        # Stimuli parameters, these are set on display in order to allow
        # easy updating after definition
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.stimuli_font = stimuli.stim_font
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.is_txt_stim = stimuli.is_txt_stim
        assert self.is_txt_stim is True, "Matrix display is a text only display"
        self.stim_length = stimuli.stim_length

        self.prompt_time = stimuli.prompt_time

        self.full_screen = full_screen

        self.symbol_set = symbol_set or alphabet()

        self.staticPeriod = static_clock

        # Set position and parameters for grid of alphabet
        self.position = stimuli.stim_pos
        self.grid_stimuli_height = .17
        self.position_increment = self.grid_stimuli_height + .05
        self.max_grid_width = 0.7
        self.stim_registry = {}

        self.start_opacity = 0.15
        self.highlight_opacity = 0.95
        self.full_grid_opacity = 0.95

        # Trigger handling
        self.first_run = True
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self.trigger_callback = TriggerCallback()
        self.marker_writer = marker_writer or NullMarkerWriter()
        self.experiment_clock = experiment_clock

        self.buffer_time = 2

        # Callback used on presentation of first stimulus.
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []
        self.space_char = space_char
        self.task_display = task_display
        self.task = task_display.build_task(self.window)

        self.scp = True  # for future row / col integration

        self.info = info
        self.info_text = info.build_info_text(window)

        # Create initial stimuli object for updating
        self.sti = stimuli.build_init_stimuli(window)

    def schedule_to(self, stimuli: list, timing: list, colors: list) -> None:
        """Schedule stimuli elements (works as a buffer).

        Args:
                stimuli(list[string]): list of stimuli text / name
                timing(list[float]): list of timings of stimuli
                colors(list[string]): list of colors
        """
        self.stimuli_inquiry = stimuli
        self.stimuli_timing = timing
        self.stimuli_colors = colors

    def do_inquiry(self) -> List[float]:
        """Do inquiry.

        Animates an inquiry of stimuli and returns a list of stimuli trigger timing.
        """
        if self.first_run:
            self._trigger_pulse()

        if self.scp:
            timing, target = self.prompt_target()
            timing.extend(self.animate_scp())
            return timing, target

        raise BciPyCoreException('Only SCP Matrix is available.')

    def build_grid(self, opacity: Optional[float] = None) -> None:
        """Build grid.

        Builds and displays a 7x4 matrix of stimuli.
        """
        pos = self.position
        for sym in self.symbol_set:
            text_stim = visual.TextStim(
                win=self.window,
                text=sym,
                opacity=opacity if opacity is not None else self.start_opacity,
                pos=pos,
                height=self.grid_stimuli_height)
            self.stim_registry[sym] = text_stim
            self.stim_registry[sym].draw()

            pos = self.increment_position(pos)

    def increment_position(self, pos: Tuple[float]) -> Tuple[float]:
        x_cordinate, y_cordinate = pos
        x_cordinate += self.position_increment
        if x_cordinate >= self.max_grid_width:
            y_cordinate -= self.position_increment
            x_cordinate = self.position[0]
        return (x_cordinate, y_cordinate)

    def prompt_target(self) -> List[float]:
        timing = []

        # select target which is first in list from the defined stimuli inquiry
        target = self.stimuli_inquiry[0]

        # cut off first two stimuli in inquiry, the target and fixation
        self.stimuli_inquiry = self.stimuli_inquiry[2:]

        # register any timing and marker callbacks
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            target)
        self.window.callOnFlip(self.marker_writer.push_marker, target)
        target_prompt = visual.TextStim(win=self.window,
                                        font=self.stimuli_font,
                                        text=f'Target: {target}',
                                        height=.25,
                                        color='Green',
                                        pos=(0, 0),
                                        wrapWidth=2,
                                        colorSpace='rgb',
                                        opacity=1,
                                        depth=-6.0)
        target_prompt.draw()
        self.draw_static()
        self.window.flip()

        core.wait(self.prompt_time)

        # append timing information
        timing.append(self.trigger_callback.timing)
        self.trigger_callback.reset()

        return timing, target

    def animate_scp(self) -> List[float]:
        """Animate SCP.

        Flashes each stimuli in stimuli_inquiry for their respective flash
        times.
        """
        timing = []
        # build grid and static
        self.build_grid(opacity=self.full_grid_opacity)
        self.draw_static()
        self.window.flip()
        core.wait(self.buffer_time)

        self.build_grid()
        self.draw_static()
        self.window.flip()
        core.wait(self.buffer_time)

        for i, sym in enumerate(self.stimuli_inquiry):

            # register any timing and marker callbacks
            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                sym)
            self.window.callOnFlip(self.marker_writer.push_marker, sym)

            # build grid and static
            self.build_grid()
            self.draw_static()

            # highlight a stimuli
            self.stim_registry[sym].setOpacity(self.highlight_opacity)
            self.stim_registry[sym].draw()
            # present stimuli and wait for self.stimuli_timing

            self.window.flip()
            core.wait(self.stimuli_timing[i])

            # reset the highlighted symbol and continue
            self.stim_registry[sym].setOpacity(self.start_opacity)
            self.stim_registry[sym].draw()

            # append timing information
            timing.append(self.trigger_callback.timing)
            self.trigger_callback.reset()

        self.build_grid()
        self.draw_static()
        self.window.flip()

        return timing

    def wait_screen(self, message: str, message_color: str) -> None:
        """Wait Screen.

        Define what happens on the screen when a user pauses a session.
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

    def draw_static(self) -> None:
        """Draw static elements in a stimulus."""
        self.task.draw()

        for info in self.info_text:
            info.draw()

    def update_task(self, text: str, color_list: List[str], pos: Tuple[float, float]) -> None:
        """Update Task.

        Update any task related display items not related to the inquiry. Ex. stimuli count 1/200.

        PARAMETERS:

        text: text for task
        color_list: list of the colors for each stimuli
        pos: position of task
        """
        self.task.text = text
        self.task.color = color_list[0]
        self.task.pos = pos

    def update_task_state(self, text: str, color_list: List[str]) -> None:
        """Update task state.

        Removes letters or appends to the right.
        Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each stimuli
        """
        self.update_task(text=text, color_list=color_list, pos=self.task.pos)

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
