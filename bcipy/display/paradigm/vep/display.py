import logging
from itertools import cycle
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional, Tuple,
                    Union)

from psychopy import core, visual  # type: ignore

import bcipy.display.components.layout as layout
from bcipy.display import (BCIPY_LOGO_PATH, Display, InformationProperties,
                           VEPStimuliProperties)
from bcipy.display.components.layout import scaled_size
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.paradigm.matrix.layout import symbol_positions
from bcipy.display.paradigm.vep.codes import (DEFAULT_FLICKER_RATES,
                                              round_refresh_rate,
                                              create_vep_codes,
                                              ssvep_to_code)
from bcipy.display.paradigm.vep.layout import BoxConfiguration, animation_path
from bcipy.display.paradigm.vep.vep_stim import VEPStim
from bcipy.helpers.clock import Clock
from bcipy.helpers.list import expanded
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.triggers import _calibration_trigger
from bcipy.helpers.symbols import BACKSPACE_CHAR

class StimTime(NamedTuple):
    """Represents the time that the given symbol was displayed"""
    symbol: str
    time: float


class StimProps(NamedTuple):
    """Represents properties of a single symbol (or stim box)"""
    symbol: Union[str, List[str]]
    duration: float
    color: str


class VEPDisplay(Display):
    """Display for VEP paradigm"""

    def __init__(self,
                 window: visual.Window,
                 experiment_clock: Clock,
                 stimuli: VEPStimuliProperties,
                 task_bar: TaskBar,
                 info: InformationProperties,
                 box_config: BoxConfiguration,
                 trigger_type: str = 'text',
                 symbol_set: Optional[List[str]] = None,
                 flicker_rates: List[int] = DEFAULT_FLICKER_RATES,
                 should_prompt_target: bool = True,
                 frame_rate: Optional[float] = None,
                 mseq_length: Optional[int] = 127):
        assert len(
            flicker_rates
        ) <= box_config.num_boxes, 'Not enough flicker rates provided'
        self.window = window
        if not frame_rate:
            frame_rate = self.window.getActualFrameRate()
            assert frame_rate, 'An accurate window frame rate could not be established'

        #check if frame_rate is within a 10 hz buffer for either 60 or 120 hz
        assert (55 <= frame_rate <= 65) or (115 <= frame_rate <= 125), \
            f"The current refresh rate is {frame_rate} hz and must be either 60 or 120 hz"

        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = round_refresh_rate(frame_rate)
        self.logger = logging.getLogger(__name__)

        self.mseq_length = mseq_length

        # number of VEP text areas
        self.vep_type: int = box_config.num_boxes

        # Stimuli parameters, these are set on display in order to allow
        # easy updating after definition
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.timing_prompt, self.timing_fixation, self.timing_stimuli = stimuli.stim_timing
        self.timing_animation = stimuli.animation_seconds
        self.stimuli_font = stimuli.stim_font
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.logger.info(self.stimuli_pos)

        self.stim_length = stimuli.stim_length
        self.should_prompt_target = should_prompt_target

        self.symbol_set = symbol_set or alphabet()
        self.sort_order = self.symbol_set.index
        self.check_configuration()

        # Build starting list of symbols
        display_container = layout.centered(parent=self.window, width_pct=0.7)
        self.starting_positions = symbol_positions(display_container,
                                                   rows=3,
                                                   columns=10)
        self.logger.info(
            f"Symbol starting positions ({str(display_container.units)} units): {self.starting_positions}"
        )

        self.starting_color = 'white'
        self.sti = self._build_inquiry_stimuli()

        self.static_clock = core.Clock()

        # Trigger handling
        self.first_stim_time: Optional[float] = None
        self.trigger_type = trigger_type
        self.experiment_clock = experiment_clock
        self._timing: List[StimTime] = []

        # Callback used on presentation of first stimulus.
        self.first_run = True
        self.first_stim_callback = lambda _sti: None

        self.task_bar = task_bar
        self.info_text = info.build_info_text(window)

        # build the VEP stimuli
        self.flicker_rates = flicker_rates
        self.logger.info(f"VEP flicker rates (hz): {flicker_rates}")
        rate = round_refresh_rate(frame_rate)
        codes = create_vep_codes(length=self.mseq_length, count=len(flicker_rates))
        print(f"Number of codes: {len(codes)}")
        print(f"Length of each code: {len(codes[0])}")
        vep_colors = [('red', 'green')] * self.vep_type
        vep_stim_size = scaled_size(0.24, self.window_size)
        self.vep = self.build_vep_stimuli(positions=box_config.positions,
                                          codes=codes,
                                          colors=cycle([('red', 'green')]),
                                          stim_size=vep_stim_size,
                                          num_squares=25)
        self.box_border_width = 4
        self.text_boxes = self._build_text_boxes(box_config)

    @property
    def box_colors(self) -> List[str]:
        """Get the colors used for boxes"""
        if self.should_prompt_target:
            [_target_color, _fixation_color, *colors] = self.stimuli_colors
        else:
            [_fixation_color, *colors] = self.stimuli_colors
        return colors

    def check_configuration(self):
        """Check that configured properties are consistent"""
        assert len(self.stimuli_pos) == self.vep_type, (
            f"stimuli position {len(self.stimuli_pos)} must be the same length as vep type {self.vep_type}"
        )

    def stim_properties(self) -> List[StimProps]:
        """Returns a tuple of (symbol, duration, and color) for each stimuli,
        including the target and fixation stim. Stimuli that represent VEP
        boxes will have a list of symbols."""
        stim_num = len(self.stimuli_inquiry)
        assert len(self.stimuli_colors
                   ) == stim_num, "Each box should have its own color"

        return [
            StimProps(symbol=props[0], duration=props[1], color='white') for props in zip(
                self.stimuli_inquiry,
                expanded(list(self.stimuli_timing), length=stim_num),
                self.stimuli_colors)
        ]

    def box_index(self, stim_groups: List[StimProps], sym: str) -> int:
        """Box index for the given symbol"""
        for i, group in enumerate(stim_groups):
            if sym in group.symbol:
                return i
        raise Exception(f"Symbol not found: {sym}")

    def do_inquiry(self) -> List[StimTime]:  # type: ignore
        """Do the inquiry."""
        self.reset_timing()

        # if this is the first run, calibrate using the trigger pulse
        if self.first_run:
            self._trigger_pulse()

        self._reset_text_boxes()
        self.reset_symbol_positions()

        if self.should_prompt_target:
            [target, _fixation, *stim] = self.stim_properties()
            if isinstance(target.symbol, str):
                self.set_stimuli_colors(stim)
                self.prompt_target(target,
                                   target_box_index=self.box_index(
                                       stim, target.symbol))
        else:
            [_fixation, *stim] = self.stim_properties()
            self.set_stimuli_colors(stim)

        self.log_inquiry(stim)
        self.animate_inquiry(stim)
        core.wait(self.timing_fixation)
        self._reset_text_boxes()
        self.stimulate(target_box_index=self.box_index(
                                       stim, target.symbol))

        # clear everything expect static stimuli
        self.draw_static()
        self.window.flip()

        return self._timing

    def starting_position(self, sym: str) -> Tuple[float, float]:
        """Get the starting position for the given symbol"""
        pos_index = self.sort_order(sym)
        return self.starting_positions[pos_index]

    def prompt_target(self,
                      target: StimProps,
                      target_box_index: Optional[int] = None) -> None:
        """Present the target for the configured length of time. Records the
        stimuli timing information.

        Parameters
        ----------
            target - (symbol, duration, color) tuple
        """
        assert isinstance(target.symbol, str), "Target must be a str"
        self.logger.info(f"Target: {target.symbol} at index {target_box_index}")

        self.window.callOnFlip(self.add_timing, 'PROMPT')

        if target_box_index is not None:
            self.highlight_target_box(target_box_index)

        self.draw_static()

        self.window.flip()
        core.wait(target.duration / 2)

    def log_inquiry(self, stimuli: List[StimProps]) -> None:
        """Log the inquiry"""
        self.logger.info(f"Inquiry: {[stim.symbol for stim in stimuli]}")

    def highlight_target_box(self, target_box_index: int) -> None:
        """Emphasize the box at the given index"""
        self.current_highlighted_box_index = target_box_index
        for i, box in enumerate(self.text_boxes):
            if i == target_box_index:
                box.borderColor = 'green'
                box.borderWidth = self.box_border_width + 15
                box.setOpacity(1)
            else:
                box.borderWidth = self.box_border_width - 2
                box.setOpacity(0.8)

    def do_fixation(self, fixation: StimProps) -> None:
        """Show all symbols before animating."""
        duration = fixation.duration or self.timing_fixation
        self.draw_static()
        self.draw_boxes()
        self.window.flip()
        core.wait(duration / 2)
        self.draw_static()
        self.draw_boxes()
        self.window.flip()
        core.wait(duration / 2)

    def animate_inquiry(self, stimuli: List[StimProps]) -> None:
        """Display the inquiry.

        Inquiry is a list of lists of strings.
        Each list contains what stimuli to display for each box defined in self.vep.
        """

        self.set_stimuli_colors(stimuli)
        self._set_inquiry(stimuli)

        # Display the inquiry with symbols in their final positions
        self.draw_boxes()
        self.draw_static()
        self.window.flip()

    def set_stimuli_colors(self, stim_groups: List[StimProps]) -> None:
        """Update the colors of the stimuli associated with each symbol to
        reflect which box it will be placed in."""
        for group in stim_groups:
            for sym in group.symbol:
                self.sti[sym].color = 'white'

    def draw_animation(self, stimuli: List[StimProps]) -> None:
        """Draw the stimuli animation.

        Parameters
        ----------
            stimuli - information about each symbol's destination.
        """
        frames = int(self.refresh_rate * self.timing_animation)
        # Compute intermediate positions
        animation_paths = {}
        for i, stim_props in enumerate(stimuli):
            end_pos = self.text_boxes[i].pos
            for sym in stim_props.symbol:
                start_pos = self.starting_position(sym)
                animation_paths[sym] = animation_path(start_pos, end_pos,
                                                      frames)
        counter = 0
        self.static_clock.reset()
        while self.static_clock.getTime() < self.timing_animation:
            for frame in range(frames):
                self.draw_boxes()
                for sym, sti in self.sti.items():
                    sti.pos = animation_paths[sym][frame]
                    sti.draw()
                self.draw_static()
                self.window.flip()
                counter += 1
        self.logger.debug(
            f"{counter} animation frames rendered in {self.timing_animation} seconds"
        )

    def draw_boxes(self) -> None:
        """Draw the text boxes under VEP stimuli."""
        for i, box in enumerate(self.text_boxes):
            if i == self.current_highlighted_box_index:
                box.borderColor = 'green'
                box.borderWidth = self.box_border_width + 10
            box.draw()

    def stimulate(self, target_box_index: int) -> None:
        """
        This is the main display function of the VEP paradigm. It is
        responsible for drawing the flickering stimuli.

        It assumes that the VEP stimuli are already constructed. These boxes
        are drawn in the order they are in the list as defined in self.vep.
        """
        self.static_clock.reset()
        self.window.callOnFlip(self.add_timing, f'STIMULATE_{target_box_index}')
        for frame in range(self.mseq_length):
            self.draw_boxes()
            for stim in self.vep:
                stim.render_frame(frame)
            # self.draw_static()
            self.window.flip()
        ended_at = self.static_clock.getTime()
        self.logger.debug(
            f"Expected stim time: {self.timing_stimuli}; actual run time: {ended_at}"
        )
        self.logger.debug(
            f"Average frame duration: {ended_at/self.timing_stimuli}")

    def draw_static(self) -> None:
        """Draw static elements in a stimulus."""
        if self.task_bar:
            self.task_bar.draw()

        for info in self.info_text:
            info.draw()

    def update_task_bar(self, text: str = ''):
        """Update any task related display items not related to the inquiry.
        Ex. stimuli count 1/200.

        Parameters
        ----------
            text - text for task
        """
        if self.task_bar:
            self.task_bar.update(text)

    def add_timing(self, stimuli: str):
        """Add a new timing entry using the stimuli as a label.

        Useful as a callback function to register a marker at the time it is
        first displayed."""
        self._timing.append(StimTime(stimuli, self.experiment_clock.getTime()))

    def reset_timing(self):
        """Reset the trigger timing."""
        self._timing = []

    def _build_inquiry_stimuli(self) -> Dict[str, visual.TextStim]:
        """Build the inquiry stimuli."""
        grid = {}
        for sym in self.symbol_set:
            pos_index = self.sort_order(sym)
            grid[sym] = visual.TextStim(win=self.window,
                                        text=sym,
                                        color='white',
                                        pos=self.starting_positions[pos_index],
                                        height=self.stimuli_height)
        return grid

    def reset_symbol_positions(self) -> None:
        """Reset the position of each symbol to its starting position"""
        #box layout
        layout = [
            ['A', 'B', 'C', 'D', 'E'],
            ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'],
            ['N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        ]

        position_index = 0
        for row in layout:
            for sym in row:
                if sym in self.symbol_set:
                    #Set position of symbol based on place in layout
                    pos = self.starting_positions[position_index]
                    self.sti[sym].pos = pos
                    position_index += 1
                    
    def build_vep_stimuli(self, positions: List[Tuple[float, float]],
                          codes: List[List[int]], colors: Iterable[Tuple[str,
                                                                         str]],
                          stim_size: Tuple[float, float],
                          num_squares: int) -> List[VEPStim]:
        """Build the VEP flashing checkerboards"""
        stim = []
        for pos, code, color in zip(positions, codes, colors):
            stim.append(
                VEPStim(self.window,
                        code,
                        color,
                        center=pos,
                        size=stim_size,
                        num_squares=num_squares))
        return stim

    def _trigger_pulse(self) -> None:
        """Trigger Pulse.

        This method uses a calibration trigger to determine any functional
            offsets needed for operation with this display. By setting the first_stim_time and searching for the
            same stimuli output to the marker stream, the offsets between these proceses can be reconciled at the
            beginning of an experiment. If drift is detected in your experiment, more frequent pulses and offset
            correction may be required.
        """
        trg = _calibration_trigger(self.experiment_clock,
                                   trigger_type=self.trigger_type,
                                   display=self.window)
        calibration_time = StimTime(*trg)
        if not self.first_stim_time:
            self.first_stim_time = calibration_time.time
            self.first_run = False

    def schedule_to(self, stimuli: List[List[Any]], timing: Optional[List[List[float]]]
                    = None, colors: Optional[List[List[str]]] = None) -> None:
        """Schedule stimuli elements (works as a buffer).
        """
        self.stimuli_inquiry = stimuli  # type: ignore
        assert timing is None or timing == self.stimuli_timing, "Timing values must match pre-configured values"
        assert colors is None or colors == self.stimuli_colors, "Colors must match the pre-configured values"

    def _build_text_boxes(
            self, box_config: BoxConfiguration) -> List[visual.TextBox2]:
        """Build the text boxes for the experiment. These are the areas into
        which the symbols are partitioned. Each text_box will have an
        associated VEP Box.
        """
        positions = box_config.positions
        size = box_config.box_size
        return [
            visual.TextBox2(win=self.window,
                            text=" ",
                            font=self.stimuli_font,
                            pos=pos,
                            units=box_config.units,
                            color='white',
                            colorSpace='rgb',
                            size=size,
                            alignment='center',
                            anchor='center',
                            borderWidth=self.box_border_width,
                            borderColor='white',
                            letterHeight=self.stimuli_height)
            for pos, color in zip(positions, cycle(self.box_colors))
        ]

    def _reset_text_boxes(self) -> None:
        """Reset text boxes.

        This method resets the text boxes to the blank state.
        """
        for text_box in self.text_boxes:
            text_box.setText(' ')
            text_box.setOpacity(1.0)
            text_box.borderWidth = self.box_border_width

    def _set_inquiry(self, stimuli: List[StimProps]) -> List[visual.TextBox2]:
        """Set the correct inquiry text for each text boxes.
        """
        #box layout
        layout = [
            ['A', 'B', 'C', 'D', 'E'],
            ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'],
            ['N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            ['MODE   ', 'SWITCH'],
            ['HER'],
            ['MAN'],
            [BACKSPACE_CHAR]
        ]
    
        for box_index, symbols in enumerate(layout):
            box = self.text_boxes[box_index]
            text = ' '.join(symbols)
            box.text = text
            box.color = 'white'
            box.borderColor = 'white'
        return self.text_boxes

    def wait_screen(self, message: str, color: str) -> None:
        """Wait Screen.

        Args:
            message(string): message to be displayed while waiting
            color(string): color of the message to be displayed
        """

        # Construct the wait message
        wait_message = visual.TextStim(win=self.window,
                                       font=self.stimuli_font,
                                       text=message,
                                       height=.1,
                                       color=color,
                                       pos=(0, -.5),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1,
                                       depth=-6.0)

        # try adding the BciPy logo to the wait screen
        try:
            wait_logo = visual.ImageStim(self.window,
                                         image=BCIPY_LOGO_PATH,
                                         pos=(0, .25),
                                         mask=None,
                                         ori=0.0)
            wait_logo.size = resize_image(BCIPY_LOGO_PATH, self.window.size, 1)
            wait_logo.draw()

        except Exception as e:
            self.logger.exception(
                f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            raise e

        # Draw and flip the screen.
        wait_message.draw()
        self.draw_static()
        self.window.flip()
