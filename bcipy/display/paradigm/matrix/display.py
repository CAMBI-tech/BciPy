"""Display for presenting stimuli in a grid.

This module provides functionality for displaying and managing matrix-style stimuli
presentations, commonly used in BCI paradigms. It handles the layout, timing, and
animation of stimuli in a grid format.
"""

import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from psychopy import core, visual

import bcipy.display.components.layout as layout
from bcipy.config import MATRIX_IMAGE_FILENAME, SESSION_LOG_FILENAME
from bcipy.core.stimuli import resize_image
from bcipy.core.symbols import alphabet, frequency_order, qwerty_order
from bcipy.core.triggers import _calibration_trigger
from bcipy.display import (BCIPY_LOGO_PATH, Display, InformationProperties,
                           StimuliProperties)
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.main import PreviewParams, init_preview_button_handler
from bcipy.display.paradigm.matrix.layout import symbol_positions

logger = logging.getLogger(SESSION_LOG_FILENAME)


class SymbolDuration(NamedTuple):
    """Represents a symbol and its associated duration to display.

    Attributes:
        symbol (str): The symbol to display.
        duration (float): Duration in seconds to display the symbol.
        color (str): Color to display the symbol in. Defaults to 'white'.
    """
    symbol: str
    duration: float
    color: str = 'white'


class MatrixDisplay(Display):
    """Matrix Display Object for Inquiry Presentation.

    Animates display objects in matrix grid common to any Matrix task.

    Attributes:
        window (visual.Window): PsychoPy window for display.
        stimuli_inquiry (List[str]): List of stimuli to present.
        stimuli_timing (List[float]): List of timing values for each stimulus.
        stimuli_colors (List[str]): List of colors for each stimulus.
        stimuli_font (str): Font to use for text stimuli.
        symbol_set (Optional[List[str]]): Set of symbols to display.
        sort_order (Callable[[str], int]): Function to determine symbol order.
        grid_stimuli_height (float): Height of grid stimuli.
        positions (Dict[int, Tuple[float, float]]): Positions for each symbol.
        grid_color (str): Default color for grid elements.
        start_opacity (float): Initial opacity for grid elements.
        highlight_opacity (float): Opacity for highlighted elements.
        full_grid_opacity (float): Opacity for full grid display.
        first_run (bool): Whether this is the first run.
        first_stim_time (Optional[float]): Time of first stimulus.
        trigger_type (str): Type of trigger to use.
        _timing (List[Tuple[str, float]]): List of timing information.
        experiment_clock (core.Clock): Clock for timing.
        task_bar (TaskBar): Task bar component.
        info_text (List[visual.TextStim]): Information text components.
        stim_registry (Dict[str, visual.TextStim]): Registry of stimuli.
        should_prompt_target (bool): Whether to prompt for target.
        preview_params (Optional[PreviewParams]): Preview configuration.
        preview_button_handler (Optional[Any]): Handler for preview buttons.
        preview_accepted (bool): Whether preview was accepted.

    Note:
        The following are recommended parameter values for matrix experiments:
        - time_fixation: 2
        - stim_pos_x: -0.6
        - stim_pos_y: 0.4
        - stim_height: 0.17
    """

    def __init__(
        self,
        window: visual.Window,
        experiment_clock: core.Clock,
        stimuli: StimuliProperties,
        task_bar: TaskBar,
        info: InformationProperties,
        rows: int = 5,
        columns: int = 6,
        width_pct: float = 0.75,
        height_pct: float = 0.8,
        trigger_type: str = 'text',
        symbol_set: Optional[List[str]] = alphabet(),
        should_prompt_target: bool = True,
        preview_config: Optional[PreviewParams] = None
    ) -> None:
        """Initialize Matrix display parameters and objects.

        Args:
            window (visual.Window): PsychoPy Window.
            experiment_clock (core.Clock): Clock used to timestamp display onsets.
            stimuli (StimuliProperties): Attributes used for inquiries.
            task_bar (TaskBar): Used for task tracking. Ex. 1/100.
            info (InformationProperties): Attributes to display informational stimuli.
            rows (int): Number of rows in the matrix. Defaults to 5.
            columns (int): Number of columns in the matrix. Defaults to 6.
            width_pct (float): Width percentage of the display. Defaults to 0.75.
            height_pct (float): Height percentage of the display. Defaults to 0.8.
            trigger_type (str): Defines the calibration trigger type. Defaults to 'text'.
            symbol_set (Optional[List[str]]): Subset of stimuli to be highlighted.
                Defaults to alphabet().
            should_prompt_target (bool): When True prompts for the target symbol.
                Defaults to True.
            preview_config (Optional[PreviewParams]): Configuration for previewing inquiries.
                Defaults to None.
        """
        self.window = window

        self.stimuli_inquiry: List[str] = []
        self.stimuli_timing: List[float] = []
        self.stimuli_colors: List[str] = []
        self.stimuli_font = stimuli.stim_font

        assert stimuli.is_txt_stim, "Matrix display is a text only display"

        self.symbol_set = symbol_set
        self.sort_order = self.build_sort_order(stimuli)
        # Set position and parameters for grid of alphabet
        self.grid_stimuli_height = stimuli.stim_height

        display_container = layout.centered(
            parent=window,
            width_pct=width_pct,
            height_pct=height_pct
        )
        self.positions = symbol_positions(
            display_container, rows, columns, symbol_set)

        self.grid_color = 'white'
        self.start_opacity = 0.15
        self.highlight_opacity = 0.95
        self.full_grid_opacity = 0.95

        # Trigger handling
        self.first_run = True
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self._timing: List[Tuple[str, float]] = []

        self.experiment_clock = experiment_clock

        self.task_bar = task_bar
        self.info_text = info.build_info_text(window)

        self.stim_registry = self.build_grid()
        self.should_prompt_target = should_prompt_target

        self.preview_params = preview_config
        self.preview_button_handler = init_preview_button_handler(
            preview_config, experiment_clock) if self.preview_enabled else None
        self.preview_accepted = True

        logger.info(
            f"Symbol positions ({display_container.units} units):\n{self.stim_positions}"
        )
        logger.info(f"Matrix center position: {display_container.center}")

    def build_sort_order(self, stimuli: StimuliProperties) -> Callable[[str], int]:
        """Build the symbol set for the display.

        Args:
            stimuli (StimuliProperties): Properties containing layout information.

        Returns:
            Callable[[str], int]: Function that returns the index for a given symbol.

        Raises:
            ValueError: If layout or symbol set is not recognized.
        """
        if stimuli.layout == 'ALP':
            if self.symbol_set:
                return self.symbol_set.index
            else:
                raise ValueError('Symbol set not defined')
        elif stimuli.layout == 'QWERTY':
            logger.info('Using QWERTY layout')
            return qwerty_order()
        elif stimuli.layout == 'FREQ':
            logger.info('Using frequency layout')
            return frequency_order()
        else:
            raise ValueError(f'Unknown layout: {stimuli.layout}')

    @property
    def stim_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get positions for each stimulus.

        Returns:
            Dict[str, Tuple[float, float]]: Dictionary mapping symbols to positions.

        Raises:
            AssertionError: If stim_registry is not initialized.
        """
        assert self.stim_registry, "stim_registry not yet initialized"
        return {
            sym: tuple(stim.pos)
            for sym, stim in self.stim_registry.items()
        }

    @property
    def preview_enabled(self) -> bool:
        """Check if inquiry preview should be enabled.

        Returns:
            bool: True if preview is enabled, False otherwise.
        """
        return bool(self.preview_params and self.preview_params.show_preview_inquiry)

    def capture_grid_screenshot(self, file_path: str) -> None:
        """Capture a screenshot of the current display.

        Args:
            file_path (str): Path where to save the screenshot.
        """
        # draw the grid and flip the window
        self.draw_grid(opacity=self.full_grid_opacity)
        tmp_task_bar = self.task_bar.current_index
        self.task_bar.current_index = 0
        self.draw_components()
        self.window.flip()

        # capture the screenshot and save it to the specified file path
        capture = self.window.getMovieFrame()
        capture.save(f'{file_path}/{MATRIX_IMAGE_FILENAME}')
        self.task_bar.current_index = tmp_task_bar

    def schedule_to(self, stimuli: List[str], timing: List[float], colors: Optional[List[str]] = None) -> None:
        """Schedule stimuli elements (works as a buffer).

        Args:
            stimuli (List[str]): List of stimuli text / name.
            timing (List[float]): List of timings of stimuli.
            colors (Optional[List[str]]): List of colors. Defaults to None.

        Raises:
            AssertionError: If lengths of stimuli and timing don't match,
                or if colors are provided but lengths don't match.
        """
        assert len(stimuli) == len(
            timing), "each stimuli must have a timing value"
        self.stimuli_inquiry = stimuli
        self.stimuli_timing = timing
        if colors:
            assert len(stimuli) == len(
                colors), "each stimuli must have a color"
            self.stimuli_colors = colors
        else:
            self.stimuli_colors = [self.grid_color] * len(stimuli)

    def symbol_durations(self) -> List[SymbolDuration]:
        """Get symbols associated with their duration for the currently configured stimuli_inquiry.

        Returns:
            List[SymbolDuration]: List of symbol durations.
        """
        return [
            SymbolDuration(*sti) for sti in zip(
                self.stimuli_inquiry, self.stimuli_timing, self.stimuli_colors)
        ]

    def add_timing(self, stimuli: str, stamp: Optional[float] = None) -> None:
        """Add a new timing entry using the stimuli as a label.

        Args:
            stimuli (str): Label for the timing entry.
            stamp (Optional[float]): Timestamp. If None, uses current time.
        """
        stamp = stamp or self.experiment_clock.getTime()
        self._timing.append((stimuli, stamp))

    def reset_timing(self) -> None:
        """Reset the trigger timing."""
        self._timing = []

    def do_inquiry(self) -> List[Tuple[str, float]]:
        """Animate an inquiry of stimuli and return timing information.

        Returns:
            List[Tuple[str, float]]: List of stimuli trigger timing.
        """
        self.preview_accepted = True
        self.reset_timing()
        symbol_durations = self.symbol_durations()

        if self.first_run:
            self._trigger_pulse()

        if self.should_prompt_target:
            [target, fixation, *stim] = symbol_durations
            self.prompt_target(target)
        else:
            [fixation, *stim] = symbol_durations

        if self.preview_enabled:
            self.preview_accepted = self.preview_inquiry(stim)

        if self.preview_accepted:
            self.animate_scp(fixation, stim)

        return self._timing

    def build_grid(self) -> Dict[str, visual.TextStim]:
        """Build the text stimuli to populate the grid.

        Returns:
            Dict[str, visual.TextStim]: Dictionary mapping symbols to text stimuli.
        """
        grid = {}
        if self.symbol_set:
            for sym in self.symbol_set:
                pos_index = self.sort_order(sym)
                pos = self.positions[pos_index]
                grid[sym] = visual.TextStim(
                    win=self.window,
                    font=self.stimuli_font,
                    text=sym,
                    color=self.grid_color,
                    opacity=self.start_opacity,
                    pos=pos,
                    height=self.grid_stimuli_height
                )
        return grid

    def draw_grid(
        self,
        opacity: float = 1,
        color: Optional[str] = 'white',
        highlight: Optional[List[str]] = None,
        highlight_color: Optional[str] = None
    ) -> None:
        """Draw the grid.

        Args:
            opacity (float): Opacity for each item in the matrix. Defaults to 1.
            color (Optional[str]): Optional color for each item in the matrix.
                Defaults to 'white'.
            highlight (Optional[List[str]]): Optional list of stim labels to be highlighted.
                Defaults to None.
            highlight_color (Optional[str]): Optional color to use for rendering the
                highlighted stim. Defaults to None.
        """
        for symbol, stim in self.stim_registry.items():
            should_highlight = highlight and (symbol in highlight)
            stim.setOpacity(
                self.highlight_opacity if should_highlight else opacity)
            stim.setColor(highlight_color
                          if highlight_color and should_highlight else color)
            stim.draw()

    def prompt_target(self, target: SymbolDuration) -> None:
        """Present the target for the configured length of time.

        Args:
            target (SymbolDuration): Target symbol and its duration.
        """
        # register any timing and marker callbacks
        self.window.callOnFlip(self.add_timing, target.symbol)
        self.draw(
            grid_opacity=self.start_opacity,
            duration=target.duration,
            highlight=[target.symbol],
            highlight_color=target.color
        )

    def preview_inquiry(self, stimuli: List[SymbolDuration]) -> bool:
        """Preview the inquiry and handle any button presses.

        Args:
            stimuli (List[SymbolDuration]): List of stimuli to highlight.

        Returns:
            bool: True if participant wants to proceed, False to reject.

        Raises:
            AssertionError: If preview is not enabled or button handler not initialized.
        """
        assert self.preview_enabled, "Preview feature not enabled."
        assert self.preview_button_handler, "Button handler must be initialized"

        handler = self.preview_button_handler
        self.window.callOnFlip(self.add_timing, 'inquiry_preview')

        self.draw_preview(stimuli)
        handler.await_response()

        if handler.has_response():
            self.add_timing(handler.response_label, handler.response_timestamp)

        if self.preview_params:
            self.draw(
                grid_opacity=self.start_opacity,
                duration=self.preview_params.preview_inquiry_isi
            )
        return handler.accept_result()

    def draw_preview(self, stimuli: List[SymbolDuration]) -> None:
        """Draw the inquiry preview by highlighting all of the symbols in the list.

        Args:
            stimuli (List[SymbolDuration]): List of stimuli to highlight.
        """
        self.draw(
            grid_opacity=self.start_opacity,
            highlight=[stim.symbol for stim in stimuli]
        )

    def draw(
        self,
        grid_opacity: float,
        grid_color: Optional[str] = None,
        duration: Optional[float] = None,
        highlight: Optional[List[str]] = None,
        highlight_color: Optional[str] = None
    ) -> None:
        """Draw all screen elements and flip the window.

        Args:
            grid_opacity (float): Opacity value to use on all grid symbols.
            grid_color (Optional[str]): Optional color to use for all grid symbols.
                Defaults to None.
            duration (Optional[float]): Optional seconds to wait after flipping the window.
                Defaults to None.
            highlight (Optional[List[str]]): Optional list of symbols to highlight in the grid.
                Defaults to None.
            highlight_color (Optional[str]): Optional color to use for rendering the
                highlighted stim. Defaults to None.
        """
        self.draw_grid(
            opacity=grid_opacity,
            color=grid_color or self.grid_color,
            highlight=highlight,
            highlight_color=highlight_color
        )
        self.draw_components()
        self.window.flip()
        if duration:
            core.wait(duration)

    def animate_scp(self, fixation: SymbolDuration, stimuli: List[SymbolDuration]) -> None:
        """Animate the given stimuli using single character presentation.

        Args:
            fixation (SymbolDuration): Fixation symbol and duration.
            stimuli (List[SymbolDuration]): List of stimuli to animate.
        """
        # Flashing the grid at full opacity is considered fixation.
        self.window.callOnFlip(self.add_timing, fixation.symbol)
        self.draw(
            grid_opacity=self.full_grid_opacity,
            grid_color=(fixation.color if self.should_prompt_target else
                        self.grid_color),
            duration=fixation.duration / 2
        )
        self.draw(
            grid_opacity=self.start_opacity,
            duration=fixation.duration / 2
        )

        for stim in stimuli:
            self.window.callOnFlip(self.add_timing, stim.symbol)
            self.draw(
                grid_opacity=self.start_opacity,
                duration=stim.duration,
                highlight=[stim.symbol],
                highlight_color=stim.color
            )
        self.draw(self.start_opacity)

    def wait_screen(self, message: str, message_color: str) -> None:
        """Display a wait screen with a message.

        Args:
            message (str): Message to display.
            message_color (str): Color of the message.
        """
        self.draw_components()

        # Construct the wait message
        wait_message = visual.TextStim(
            win=self.window,
            font=self.stimuli_font,
            text=message,
            height=.1,
            color=message_color,
            pos=(0, -.5),
            wrapWidth=2
        )

        # try adding the BciPy logo to the wait screen
        try:
            wait_logo = visual.ImageStim(
                self.window,
                image=BCIPY_LOGO_PATH,
                pos=(0, .25),
                mask=None,
                ori=0.0
            )
            wait_logo.size = resize_image(BCIPY_LOGO_PATH, self.window.size, 1)
            wait_logo.draw()

        except Exception as e:
            logger.exception(
                f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            raise e

        # Draw and flip the screen.
        wait_message.draw()
        self.window.flip()

    def draw_static(self) -> None:
        """Draw static elements in a stimulus."""
        self.draw_grid(self.start_opacity)
        self.draw_components()

    def draw_components(self) -> None:
        """Draw task bar and info text components."""
        if self.task_bar:
            self.task_bar.draw()

        for info in self.info_text:
            info.draw()

    def update_task_bar(self, text: str = '') -> None:
        """Update task related display items.

        Args:
            text (str): Text for task. Defaults to ''.
        """
        if self.task_bar:
            self.task_bar.update(text)

    def _trigger_pulse(self) -> None:
        """Send a calibration trigger pulse.

        This method uses a calibration trigger to determine any functional
        offsets needed for operation with this display. By setting the first_stim_time
        and searching for the same stimuli output to the marker stream, the offsets
        between these processes can be reconciled at the beginning of an experiment.
        If drift is detected in your experiment, more frequent pulses and offset
        correction may be required.
        """
        calibration_time = _calibration_trigger(
            self.experiment_clock,
            trigger_type=self.trigger_type,
            display=self.window
        )

        # set the first stim time if not present and first_run to False
        if not self.first_stim_time:
            self.first_stim_time = calibration_time[-1]
            self.first_run = False
