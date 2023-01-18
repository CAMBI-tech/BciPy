"""Display for presenting stimuli in a grid."""
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging

from psychopy import visual, core

from bcipy.display import Display, StimuliProperties, TaskDisplayProperties, InformationProperties, BCIPY_LOGO_PATH
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.triggers import _calibration_trigger
from bcipy.helpers.task import alphabet


class SymbolDuration(NamedTuple):
    """Represents a symbol and its associated duration to display"""
    symbol: str
    duration: float
    color: str = 'white'


class MatrixDisplay(Display):
    """Matrix Display Object for Inquiry Presentation.

    Animates display objects in matrix grid common to any Matrix task.

    NOTE: The following are recommended parameter values for matrix experiments:

    time_fixation: 2
    stim_pos_x: -0.6
    stim_pos_y: 0.4
    stim_height: 0.1
    """

    def __init__(
            self,
            window: visual.Window,
            experiment_clock: core.Clock,
            stimuli: StimuliProperties,
            task_display: TaskDisplayProperties,
            info: InformationProperties,
            trigger_type: str = 'text',
            symbol_set: Optional[List[str]] = None,
            should_prompt_target: bool = True):
        """Initialize Matrix display parameters and objects.

        PARAMETERS:
        ----------
        # Experiment
        window(visual.Window): PsychoPy Window
        experiment_clock(core.Clock): Clock used to timestamp display onsets

        # Stimuli
        stimuli(StimuliProperties): attributes used for inquiries

        # Task
        task_display(TaskDisplayProperties): attributes used for task tracking. Ex. 1/100

        # Info
        info(InformationProperties): attributes to display informational stimuli alongside task and inquiry stimuli.

        trigger_type(str) default 'image': defines the calibration trigger type for the display at the beginning of any
            task. This will be used to reconcile timing differences between acquisition and the display.
        symbol_set default = none : subset of stimuli to be highlighted during an inquiry
        should_prompt_target(bool): when True prompts for the target symbol. Assumes that this is
            the first symbol of each inquiry. For example: [target, fixation, *stim].
        """
        self.window = window

        self.logger = logging.getLogger(__name__)

        self.stimuli_inquiry = []
        self.stimuli_timing = []
        self.stimuli_colors = []
        self.stimuli_font = stimuli.stim_font

        assert stimuli.is_txt_stim, "Matrix display is a text only display"

        self.symbol_set = symbol_set or alphabet()

        # Set position and parameters for grid of alphabet
        self.position = stimuli.stim_pos
        self.grid_stimuli_height = .17
        self.position_increment = self.grid_stimuli_height + .05
        self.max_grid_width = 0.7

        self.grid_color = 'white'
        self.start_opacity = 0.15
        self.highlight_opacity = 0.95
        self.full_grid_opacity = 0.95

        # Trigger handling
        self.first_run = True
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self._timing = []

        self.experiment_clock = experiment_clock

        self.task = task_display.build_task(self.window)
        self.info_text = info.build_info_text(window)

        self.stim_registry = self.build_grid()
        self.should_prompt_target = should_prompt_target

    def schedule_to(self, stimuli: list, timing: list, colors: list) -> None:
        """Schedule stimuli elements (works as a buffer).

        Args:
                stimuli(list[string]): list of stimuli text / name
                timing(list[float]): list of timings of stimuli
                colors(list[string]): list of colors
        """
        assert len(stimuli) == len(
            timing), "each stimuli must have a timing value"
        self.stimuli_inquiry = stimuli
        self.stimuli_timing = timing
        if colors:
            assert len(stimuli) == len(colors), "each stimuli must have a color"
            self.stimuli_colors = colors
        else:
            self.stimuli_colors = [self.grid_color] * len(stimuli)

    def symbol_durations(self) -> List[SymbolDuration]:
        """Symbols associated with their duration for the currently configured
        stimuli_inquiry."""
        return [
            SymbolDuration(*sti)
            for sti in zip(self.stimuli_inquiry, self.stimuli_timing, self.stimuli_colors)
        ]

    def add_timing(self, stimuli: str):
        """Add a new timing entry using the stimuli as a label.

        Useful as a callback function to register a marker at the time it is
        first displayed."""
        self._timing.append([stimuli, self.experiment_clock.getTime()])

    def reset_timing(self):
        """Reset the trigger timing."""
        self._timing = []

    def do_inquiry(self) -> List[float]:
        """Animates an inquiry of stimuli and returns a list of stimuli trigger timing."""
        self.reset_timing()
        symbol_durations = self.symbol_durations()

        if self.first_run:
            self._trigger_pulse()

        if self.should_prompt_target:
            [target, fixation, *stim] = symbol_durations
            self.prompt_target(target)
        else:
            [fixation, *stim] = symbol_durations

        self.animate_scp(fixation, stim)

        return self._timing

    def build_grid(self) -> Dict[str, visual.TextStim]:
        """Build grid.

        Builds a 7x4 matrix of stimuli.
        """
        grid = {}
        pos = self.position
        for sym in self.symbol_set:
            grid[sym] = visual.TextStim(win=self.window,
                                        text=sym,
                                        color=self.grid_color,
                                        opacity=self.start_opacity,
                                        pos=pos,
                                        height=self.grid_stimuli_height)
            pos = self.increment_position(pos)
        return grid

    def increment_position(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Computes the position of the next symbol in the matrix."""
        x_coordinate, y_coordinate = pos
        x_coordinate += self.position_increment
        if x_coordinate >= self.max_grid_width:
            y_coordinate -= self.position_increment
            x_coordinate = self.position[0]
        return (x_coordinate, y_coordinate)

    def draw_grid(self,
                  opacity: float = 1,
                  color: Optional[str] = 'white',
                  highlight: Optional[str] = None,
                  highlight_color: Optional[str] = None):
        """Draw the grid.

        Parameters
        ----------
            opacity - opacity for each item in the matrix
            color - optional color for each item in the matrix
            highlight - optional stim label for the item to be highlighted
                (rendered using the highlight_opacity).
            highlight_color - optional color to use for rendering the
              highlighted stim.
        """
        for symbol, stim in self.stim_registry.items():
            stim.setOpacity(self.highlight_opacity if highlight ==
                            symbol else opacity)
            stim.setColor(highlight_color if highlight_color and
                          highlight == symbol else color)
            stim.draw()

    def prompt_target(self, target: SymbolDuration) -> float:
        """Present the target for the configured length of time. Records the
        stimuli timing information.

        Parameters
        ----------
            target - (symbol, duration) tuple
        """
        # register any timing and marker callbacks
        self.window.callOnFlip(self.add_timing, target.symbol)
        self.draw(grid_opacity=self.start_opacity,
                  duration=target.duration,
                  highlight=target.symbol,
                  highlight_color=target.color)

    def draw(self,
             grid_opacity: float,
             grid_color: Optional[str] = None,
             duration: Optional[float] = None,
             highlight: Optional[str] = None,
             highlight_color: Optional[str] = None):
        """Draw all screen elements and flip the window.

        Parameters
        ----------
            grid_opacity - opacity value to use on all grid symbols
            grid_color - optional color to use for all grid symbols
            duration - optional seconds to wait after flipping the window.
            highlight - optional symbol to highlight in the grid.
            highlight_color - optional color to use for rendering the
              highlighted stim.
        """
        self.draw_grid(opacity=grid_opacity,
                       color=grid_color or self.grid_color,
                       highlight=highlight,
                       highlight_color=highlight_color)
        self.draw_static()
        self.window.flip()
        if duration:
            core.wait(duration)

    def animate_scp(self, fixation: SymbolDuration,
                    stimuli: List[SymbolDuration]):
        """Animate the given stimuli using single character presentation.

        Flashes each stimuli in stimuli_inquiry for their respective flash
        times and records the timing information.
        """

        # Flashing the grid at full opacity is considered fixation.
        self.window.callOnFlip(self.add_timing, fixation.symbol)
        self.draw(grid_opacity=self.full_grid_opacity,
                  grid_color=fixation.color,
                  duration=fixation.duration / 2)
        self.draw(grid_opacity=self.start_opacity,
                  duration=fixation.duration / 2)

        for stim in stimuli:
            self.window.callOnFlip(self.add_timing, stim.symbol)
            self.draw(grid_opacity=self.start_opacity,
                      duration=stim.duration,
                      highlight=stim.symbol,
                      highlight_color=stim.color)
        self.draw(self.start_opacity)

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
                                       wrapWidth=2)

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
