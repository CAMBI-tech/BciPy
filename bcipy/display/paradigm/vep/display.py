import logging
from itertools import cycle
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from psychopy import core, visual
from psychopy.visual.shape import ShapeStim

import bcipy.display.components.layout as layout
from bcipy.display import (BCIPY_LOGO_PATH, Display, InformationProperties,
                           VEPStimuliProperties)
from bcipy.display.components.layout import envelope, scaled_size
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.paradigm.matrix.layout import symbol_positions
from bcipy.display.paradigm.vep.layout import BoxConfiguration, checkerboard
from bcipy.helpers.clock import Clock
from bcipy.helpers.list import expanded
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.triggers import _calibration_trigger


def create_vep_codes(length=32, count=4) -> List[List[int]]:
    """Create a list of random VEP codes.

    length - how many bits in each code. This should be greater than or equal to the refresh rate
        if using these to flicker. For example, if the refresh rate is 60Hz, then the length should
        be at least 60.
    count - how many codes to generate, each will be unique.
    """
    np.random.seed(1)
    return [np.random.randint(2, size=length) for _ in range(count)]

class StimProps(NamedTuple):
    """Represents properties of a single symbol (or stim box)"""
    symbol: Union[str, List[str]]
    duration: float
    color: str

class VEPStim:
    """Represents a checkerboard of squares that can be flashed at a given
    rate. Flashing is accomplished by inverting the colors of each square.

    Parameters
    ----------
        layout - used to build the stimulus
        code - A list of integers representing the VEP code for each box
        colors - tuple of colors for the checkerboard pattern
        center - center position of the checkerboard
        size - size of the checkerboard, in layout units
        num_squares - number of squares in the checkerboard
    """

    def __init__(self,
                 win: visual.Window,
                 code: List[int],
                 colors: Tuple[str, str],
                 center: Tuple[float, float],
                 size: Tuple[float, float],
                 num_squares: int = 4):
        self.window = win
        self.code = code

        squares = checkerboard(squares=num_squares,
                               colors=colors,
                               center=center,
                               board_size=size)
        board_boundary = envelope(pos=center, size=size)

        frame1_holes = []
        frame2_holes = []
        for square in squares:
            square_boundary = envelope(pos=square.pos, size=square.size)
            # squares define the holes in the polygon
            if square.color == colors[0]:
                frame2_holes.append(square_boundary)
            elif square.color == colors[1]:
                frame1_holes.append(square_boundary)

        # Checkerboard is represented as a polygon with holes, backed by a
        # simple square with the alternating color.
        # This technique renders more efficiently and scales better than using
        # separate shapes (Rect or Gradient) for each square.
        background = ShapeStim(self.window,
                               lineColor=colors[1],
                               fillColor=colors[1],
                               vertices=board_boundary)
        self.on_stim = [
            background,
            # polygon with holes
            ShapeStim(self.window,
                      lineWidth=0,
                      fillColor=colors[0],
                      closeShape=True,
                      vertices=[board_boundary, *frame1_holes])
        ]
        self.off_stim = [
            background,
            # polygon with holes
            ShapeStim(self.window,
                      lineWidth=0,
                      fillColor=colors[0],
                      closeShape=True,
                      vertices=[board_boundary, *frame2_holes])
        ]

    def render_frame(self, frame: int) -> None:
        """Render a given frame number, where frame refers to a code index"""
        if self.code[frame] == 1:
            self.frame_on()
        else:
            self.frame_off()

    def frame_on(self) -> None:
        """Each square is set to a starting color and draw."""
        for stim in self.on_stim:
            stim.draw()

    def frame_off(self) -> None:
        """Invert each square from its starting color and draw."""
        for stim in self.off_stim:
            stim.draw()


class VEPDisplay(Display):
    """Display for VEP paradigm"""

    def __init__(
            self,
            window: visual.Window,
            experiment_clock: Clock,
            stimuli: VEPStimuliProperties,
            task_bar: TaskBar,
            info: InformationProperties,
            trigger_type: str = 'text',
            symbol_set: Optional[List[str]] = None,
            codes: List[int] = None,
            box_config: BoxConfiguration = None,
            should_prompt_target: bool = True):
        self.window = window
        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = round(window.getActualFrameRate())
        self.logger = logging.getLogger(__name__)

        # number of VEP text areas
        self.vep_type = box_config.num_boxes

        # Stimuli parameters, these are set on display in order to allow
        # easy updating after definition
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.timing_prompt, self.timing_fixation, self.timing_stimuli = stimuli.stim_timing
        self.timing_animation = 2
        self.stimuli_font = stimuli.stim_font
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos

        self.stim_length = stimuli.stim_length
        self.logger.info(self.stimuli_pos)
        self.check_configuration()

        self.symbol_set = symbol_set or alphabet()
        self.sort_order = self.symbol_set.index

        # Build starting list of symbols
        display_container = layout.centered(parent=self.window, width_pct=0.7)
        self.starting_positions = symbol_positions(display_container,
                                                   rows=3,
                                                   columns=10)
        self.logger.info(
            f"Symbol starting positions ({display_container.units} units): {self.starting_positions}"
        )

        self.fixation = self._build_fixation()
        self.starting_color = 'white'
        self.sti = self._build_inquiry_stimuli()

        self.static_clock = core.Clock()

        # Trigger handling
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self.experiment_clock = experiment_clock
        self._timing = []

        # Callback used on presentation of first stimulus.
        self.first_run = True
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []

        self.task_bar = task_bar
        self.info_text = info.build_info_text(window)

        # build the VEP stimuli
        if not codes:
            codes = create_vep_codes(length=self.refresh_rate, count=self.vep_type)
        vep_colors = [('white', 'black'), ('red', 'green'), ('blue', 'yellow'), ('orange', 'green')]
        vep_stim_size = scaled_size(0.2, self.window_size)
        self.vep = self.build_vep_stimuli(positions=box_config.positions,
                                          codes=codes,
                                          colors=cycle(vep_colors),
                                          stim_size=vep_stim_size,
                                          num_squares=25)

        self.text_boxes = self._build_text_boxes(box_config)
        self.should_prompt_target = should_prompt_target

    def check_configuration(self):
        """Check that configured properties are consistent"""
        assert len(
            self.stimuli_pos) == self.vep_type, (
                f"stimuli position {len(self.stimuli_pos)} must be the same length as vep type {self.vep_type}")

    def do_fixation(self, fixation: StimProps) -> None:
        """Draw fixation cross"""
        duration = fixation.duration or self.timing_fixation
        self.fixation.text = fixation.symbol
        self.fixation.color = fixation.color
        self.fixation.draw()
        self.draw_static()
        self.window.flip()
        core.wait(duration)

    def stim_properties(self) -> List[StimProps]:
        """Returns a tuple of (symbol, duration, and color) for each stimuli,
        including the target and fixation stim. Stimuli that represent VEP
        boxes will have a list of symbols."""
        stim_num = len(self.stimuli_inquiry)
        assert len(self.stimuli_colors) == stim_num, "Each box should have its own color"

        return [
            StimProps(*props)
            for props in zip(self.stimuli_inquiry,
                             expanded(list(self.stimuli_timing), length=stim_num),
                             self.stimuli_colors)
        ]

    def do_inquiry(self) -> List[float]:
        """Do the inquiry."""
        self.reset_timing()

        # if this is the first run, calibrate using the trigger pulse
        if self.first_run:
            self._trigger_pulse()

        if self.should_prompt_target:
            [target, fixation, *stim] = self.stim_properties()
            self.prompt_target(target)
        else:
            [fixation, *stim] = self.stim_properties()

        # fixation --> animation / prompting --> VEP stimulate
        self.do_fixation(fixation)
        self.animate_inquiry(stim)
        self.stimulate()

        # clear everything expect static stimuli
        self.draw_static()
        self.window.flip()

        return self._timing


    def prompt_target(self, target: StimProps) -> float:
        """Present the target for the configured length of time. Records the
        stimuli timing information.

        Parameters
        ----------
            target - (symbol, duration, color) tuple
        """
        # register any timing and marker callbacks
        self.window.callOnFlip(self.add_timing, target.symbol)
        stim = visual.TextStim(self.window,
                               text=target.symbol,
                               color=target.color,
                               height=self.stimuli_height,
                               pos=[0, 0])
        stim.draw()
        self.draw_static()
        self.window.flip()
        core.wait(target.duration)

    def animate_inquiry(self, stimuli: List[StimProps]) -> None:
        """Display the inquiry.

        Inquiry is a list of lists of strings.
        Each list contains what stimuli to display for each box defined in self.vep.
        """
        self.window.callOnFlip(self.add_timing, 'VEP_INQ_ANIMATION')

        self._reset_text_boxes()
        self.reset_symbol_positions()
        self.set_stimuli_colors(stimuli)
        self.draw_animation()
        self._set_inquiry(stimuli)

        self.window.callOnFlip(self.add_timing, 'VEP_INQUIRY')
        # display the inquiry
        self.draw_boxes()
        self.draw_static()
        self.window.flip()
        core.wait(self.timing_prompt)

    def set_stimuli_colors(self, stim_groups: List[StimProps]) -> None:
        """Update the colors of the stimuli associated with each symbol to
        reflect which box it will be placed in."""
        for group in stim_groups:
            for sym in group.symbol:
                self.sti[sym].color = group.color

    def draw_animation(self) -> None:
        """Draw the stimuli animation.

        TODO: animate the symbols moving to their end positions
        - determine the number of frames in self.timing_animation seconds
        - calculate end positions for each symbol (using symbol_positions with each box as the layout)
        - compute the intermediate positions (line trajectory of each symbol)
        - update position on each tick
        """
        self.static_clock.reset()
        while self.static_clock.getTime() < self.timing_animation:
            self.draw_boxes()
            for sti in self.sti.values():
                sti.draw()
            self.draw_static()
            self.window.flip()

    def draw_boxes(self) -> None:
        """Draw the text boxes under VEP stimuli."""
        for box in self.text_boxes:
            box.draw()

    def stimulate(self) -> None:
        """
        This is the main display function of the VEP paradigm. It is
        responsible for drawing the flickering stimuli.

        It assumes that the VEP stimuli are already constructed. These boxes
        are drawn in the order they are in the list as defined in self.vep.
        """
        self.window.callOnFlip(self.add_timing, 'VEP_STIMULATE')
        self.static_clock.reset()
        while self.static_clock.getTime() < self.timing_stimuli:
            for frame in range(self.refresh_rate):
                for stim in self.vep:
                    stim.render_frame(frame)

                self.draw_boxes()
                # self.draw_static()
                self.window.flip()
        ended_at = self.static_clock.getTime()
        self.logger.debug(
            f"Expected stim time: {self.timing_stimuli}; actual run time: {ended_at}"
        )
        self.logger.debug(f"Average frame duration: {ended_at/self.timing_stimuli}")


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
        self._timing.append([stimuli, self.experiment_clock.getTime()])

    def reset_timing(self):
        """Reset the trigger timing."""
        self._timing = []

    def _build_fixation(self) -> visual.TextStim:
        """Build the fixation stim"""
        return visual.TextStim(self.window,
                               text='+',
                               color='red',
                               height=self.stimuli_height,
                               pos=[0, 0])

    def _build_inquiry_stimuli(self) -> List[visual.TextStim]:
        """Build the inquiry stimuli."""
        grid = {}
        for sym in self.symbol_set:
            pos_index = self.sort_order(sym)
            grid[sym] = visual.TextStim(win=self.window,
                                        text=sym,
                                        color=self.starting_color,
                                        pos=self.starting_positions[pos_index],
                                        height=self.stimuli_height)
        return grid

    def reset_symbol_positions(self) -> None:
        """Reset the position of each symbol to its starting position"""
        for sym in self.symbol_set:
            pos_index = self.sort_order(sym)
            self.sti[sym].pos = self.starting_positions[pos_index]

    def build_vep_stimuli(self,
                          positions: List[Tuple[float, float]],
                          codes: List[int],
                          colors: List[Tuple[str]],
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
        calibration_time = _calibration_trigger(
            self.experiment_clock,
            trigger_type=self.trigger_type,
            display=self.window)

        # set the first stim time if not present and first_run to False
        if not self.first_stim_time:
            self.first_stim_time = calibration_time[-1]
            self.first_run = False

    def schedule_to(self, stimuli: List[List[str]], timing: List[List[float]]
                    = None, colors: List[List[str]] = None) -> None:
        """Schedule stimuli elements (works as a buffer).
        """
        self.stimuli_inquiry = stimuli
        assert timing is None or timing == self.stimuli_timing, "Timing values must match pre-configured values"
        assert colors is None or colors == self.stimuli_colors, "Colors must match the pre-configured values"

    def _build_text_boxes(self, box_config: BoxConfiguration) -> List[visual.TextBox2]:
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
                            color=color,
                            colorSpace='rgb',
                            size=size,
                            alignment='center',
                            anchor='center',
                            borderWidth=2,
                            borderColor=color,
                            letterHeight=self.stimuli_height)
            for pos, color in zip(positions, cycle(self.stimuli_colors))
        ]

    def _reset_text_boxes(self) -> None:
        """Reset text boxes.

        This method resets the text boxes to the blank state.
        """
        for text_box in self.text_boxes:
            text_box.setText(' ')

    def _set_inquiry(self, stimuli: List[StimProps]) -> List[visual.TextBox2]:
        """Set the correct inquiry text for each text boxes.
        """
        for i, sti in enumerate(stimuli):
            box = self.text_boxes[i]
            text = ' '.join(sti.symbol)
            box.text = text
            box.color = sti.color
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
        self.draw_static()
        self.window.flip()
