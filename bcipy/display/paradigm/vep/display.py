import logging
from typing import List, Tuple, Optional
from bcipy.display import (
    BCIPY_LOGO_PATH,
    Display,
    InformationProperties,
    VEPStimuliProperties,
    TaskDisplayProperties,
)
import numpy as np
from psychopy import visual, core
from bcipy.helpers.clock import Clock
from bcipy.helpers.task import SPACE_CHAR
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger
from bcipy.helpers.stimuli import resize_image, get_fixation


def create_vep_codes(length=32, count=4) -> List[List[int]]:
    """Create a list of random VEP codes.

    length - how many bits in each code. This should be greater than or equal to the refresh rate
        if using these to flicker. For example, if the refresh rate is 60Hz, then the length should
        be at least 60.
    count - how many codes to generate, each will be unique.
    """
    np.random.seed(1)
    return [np.random.randint(2, size=length) for _ in range(count)]


class VEPBox:
    """A class to represent a VEP box.

    Attributes:
        flicker: A list of two visual.GratingStim objects, one for on and one for off.
        code: A list of integers representing the VEP code for this box.
    """

    def __init__(self, flicker: List[Tuple[visual.GratingStim, visual.GratingStim]], code: List[int]) -> None:
        self.flicker = flicker
        self.code = code

    def frame_on(self) -> None:
        """Frame On. Draw the frame for the first grating stim on the screen."""
        self.flicker[0].draw()

    def frame_off(self) -> None:
        """Frame Off. Draw the frame for the second grating stim on the screen."""
        self.flicker[1].draw()


class VEPDisplay(Display):

    def __init__(
            self,
            window: visual.Window,
            experiment_clock: Clock,
            stimuli: VEPStimuliProperties,
            task_display: TaskDisplayProperties,
            info: InformationProperties,
            trigger_type: str = 'text',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False,
            codes: List[int] = None):
        self.window = window
        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = round(window.getActualFrameRate())

        self.logger = logging.getLogger(__name__)

        # Stimuli parameters, these are set on display in order to allow
        #  easy updating after defintion
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.stimuli_font = stimuli.stim_font
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.logger.info(self.stimuli_pos)
        self.is_txt_stim = stimuli.is_txt_stim
        self.stim_length = stimuli.stim_length
        self.sti = []
        self.fixation = visual.TextStim(
            self.window,
            text=get_fixation(True),
            color='red',
            height=self.stimuli_height,
            units='height',
            pos=[0, 0])

        self.full_screen = full_screen

        self.static_clock = core.Clock()

        # Trigger handling
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self.trigger_callback = TriggerCallback()
        self.experiment_clock = experiment_clock

        # Callback used on presentation of first stimulus.
        self.first_run = True
        self.first_stim_callback = lambda _sti: None
        self.size_list_sti = []

        # Task parameters
        self.space_char = space_char
        self.task_display = task_display
        self.task = task_display.build_task(self.window)

        # Information parameters
        self.info = info
        self.info_text = info.build_info_text(window)

        self.colors = [('white', 'black'), ('red', 'green'), ('blue', 'yellow'), ('orange', 'green')]
        if not codes:
            self.codes = create_vep_codes(length=self.refresh_rate, count=4)
        else:
            self.codes = codes

        # build the VEP stimuli
        self.vep = []
        self.text_boxes = []
        self.vep_type = 4
        assert self.vep_type == 4, 'Only 4 stimuli are supported in VEP display at this time!'
        self.vep_box_size = .1

        self.animation_duration = 2

    def do_fixation(self) -> None:
        # draw fixation cross
        self.fixation.draw()
        self.draw_static()
        self.window.flip()
        core.wait(self.stimuli_timing[1])

    def do_inquiry(self) -> List[float]:
        """Do the inquiry."""

        timing = []

        # if this is the first run, calibrate using the trigger pulse and setup the VEP and text boxes
        if self.first_run:
            self._trigger_pulse()
            self._build_vep_corner_stimuli(self.stimuli_pos, self.codes, self.colors, self.vep_box_size)
            self._build_text_boxes()

        # fixation --> animation / prompting --> VEP stimulate
        self.do_fixation()
        timing += self.animate_inquiry()
        timing += self.stimulate()

        # clear everyting expect static stimuli
        self.draw_static()
        self.window.flip()

        return timing

    def animate_inquiry(self) -> List[float]:
        """Display the inquiry.

        Inquiry is a list of lists of strings.
        Each list contains what stimuli to display for each box defined in self.vep.
        """
        timing = []
        self.trigger_callback.reset()
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'VEP_INQ_ANIMATION')

        self._reset_text_boxes()
        self._build_inquiry_stimuli()

        self.static_clock.reset()
        while self.static_clock.getTime() < self.animation_duration:
            self.draw_boxes()
            self.draw_animation()
            self.draw_static()
            self.window.flip()
        timing += self.trigger_callback.timing

        self._set_inquiry()
        self.trigger_callback.reset()
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'VEP_INQUIRY')
        # display the inquiry
        self.static_clock.reset()
        while self.static_clock.getTime() < self.stimuli_timing[0]:
            self.draw_boxes()
            self.draw_static()
            self.window.flip()

        timing += self.trigger_callback.timing

        return timing

    def draw_animation(self) -> None:
        """Draw the stimuli animation."""
        for sti in self.sti:
            sti.draw()

    def draw_boxes(self) -> None:
        """Draw the text boxes under VEP stimuli."""
        for box in self.text_boxes:
            box.draw()

    def stimulate(self) -> List[float]:
        """Stimulate.

        This is the main display function of the VEP paradigm. It is responsible for drawing the flickering stimuli.

        It assumes that the VEP stimuli are already constructed using the _build_vep_corner_stimuli function.
        These boxes are drawn in the order they are in the list as defined in self.vep."""
        self.trigger_callback.reset()
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'VEP_STIMULATE')
        self.static_clock.reset()
        while self.static_clock.getTime() < self.stimuli_timing[-1]:
            for frame in range(self.refresh_rate):
                for box in self.vep:
                    if box.code[frame] == 1:
                        box.frame_on()
                    else:
                        box.frame_off()

                # flicker!
                self.draw_static()
                self.window.flip()

        return self.trigger_callback.timing

    def draw_static(self) -> None:
        """Draw static elements for the display."""
        self.task.draw()
        for info in self.info_text:
            info.draw()

    def update_task(self, text: str, color: str, pos: Optional[Tuple] = None) -> None:
        """Update the task display which is displayed using the draw_static function."""
        self.task.text = text
        self.task.color = color
        if pos:
            self.task.pos = pos

    def _build_inquiry_stimuli(self) -> None:
        """Build the inquiry stimuli."""
        all_letters = []
        all_colors = []
        self.sti = []
        for letters, colors in zip(self.stimuli_inquiry, self.stimuli_colors):
            for letter in letters:
                all_letters.append(letter)
                all_colors.append(colors[0])
        pos = (0, 0)
        for letter, color in zip(all_letters, all_colors):
            pos = (pos[0] + self.stimuli_height, pos[1])
            self.sti.append(
                visual.TextStim(self.window, text=letter, color=color, pos=pos, height=self.stimuli_height))

        return self.sti

    def _build_vep_corner_stimuli(
            self,
            pos: Tuple[float, float],
            codes: List[int],
            colors: List[Tuple[str]],
            box_size: float) -> List[visual.GratingStim]:
        """Build the corner stimuli for the VEP.

        Args:
            window: psychopy window object
        """
        corner_stim = []
        for pos, code, color in zip(pos, codes, colors):
            corner_stim += self._build_2x2_vep_grid(pos, box_size, color, code)

        return corner_stim

    def _build_1x1_vep_grid(
            self,
            position: Tuple[float],
            size: float,
            color: str,
            code: List[int]) -> List[visual.GratingStim]:
        """Build a 1x1 VEP grid.

        Args:
            position: position of the flicking box
            size: size of the flicking box
            color: color of the flicking box
            code: code to be used with the flicking box

        Returns:
            list of visual.GratingStim objects

        """
        assert len(code) >= self.refresh_rate, 'Code must be longer than refresh rate'
        pattern1 = visual.GratingStim(win=self.window, name=f'1x1-1-{position}', units='height',
                                      tex=None, mask=None,
                                      ori=0, pos=position, size=size, sf=1, phase=0.0,
                                      color=color[0], colorSpace='rgb', opacity=1,
                                      texRes=256, interpolate=True, depth=-1.0)
        pattern2 = visual.GratingStim(win=self.window, name=f'1x1-2-{position}', units='height',
                                      tex=None, mask=None,
                                      ori=0, pos=position, size=size, sf=1, phase=0,
                                      color=color[1], colorSpace='rgb', opacity=1,
                                      texRes=256, interpolate=True, depth=-2.0)
        box = [VEPBox(flicker=[pattern1, pattern2], code=code)]
        self.vep += box
        return box

    def _build_2x2_vep_grid(self,
                            position: Tuple[float],
                            size: float,
                            color: str,
                            code: List[int]) -> List[visual.GratingStim]:
        """Build a 2x2 VEP grid.

        Args:
            position: position of the flicking box
            size: size of the flicking box
            color: color of the flicking box
            code: code to be used with the flicking box

        Returns:
            list of visual.GratingStim objects
        """

        assert len(code) >= self.refresh_rate, 'Code must be longer than refresh rate'
        starting_position = position
        positions = [
            starting_position,  # bottom left
            [starting_position[0] + size, starting_position[1]],  # bottom right
            [starting_position[0] + size, starting_position[1] + size],  # top right
            [starting_position[0], starting_position[1] + size],  # top left
        ]
        box = []
        for pos in positions:
            pattern1 = visual.GratingStim(win=self.window, name=f'2x2-1-{pos}', units='height',
                                          tex=None, mask=None,
                                          ori=0, pos=pos, size=size, sf=1, phase=0.0,
                                          color=color[0], colorSpace='rgb', opacity=1,
                                          texRes=256, interpolate=True, depth=-1.0)
            pattern2 = visual.GratingStim(win=self.window, name=f'2x2-2-{pos}', units='height',
                                          tex=None, mask=None,
                                          ori=0, pos=pos, size=size, sf=1, phase=0,
                                          color=color[1], colorSpace='rgb', opacity=1,
                                          texRes=256, interpolate=True, depth=-2.0)
            color = (color[1], color[0])
            box += [VEPBox(flicker=[pattern1, pattern2], code=code)]

        self.vep += box
        return box

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

    def schedule_to(self, stimuli: List[List[str]], timing: List[List[float]], colors: List[List[str]]) -> None:
        """Schedule stimuli elements (works as a buffer).
        """
        self.stimuli_inquiry = stimuli
        self.stimuli_timing = timing
        self.stimuli_colors = colors

    def _build_text_boxes(self) -> List[visual.TextBox2]:
        """Build the text boxes for the experiment.

        This assumes that vep_type is 4 using a 2x2 grid. Additional configurations can be added in the future.
        """
        # generate the inquiry stimuli
        assert len(
            self.stimuli_colors) == self.vep_type, (
                f"stmuli colors {len(self.stimuli_colors)} must be the same length as vep type {self.vep_type}")
        assert len(
            self.stimuli_pos) == self.vep_type, (
                f"stmuli position {len(self.stimuli_pos)} must be the same length as vep type {self.vep_type}")

        self.text_boxes = []
        inc_q = int(self.vep_type / 2)
        size = self.vep_box_size * inc_q
        inc = 0  # get the last box in the grating stim per vep
        for color in self.stimuli_colors:
            text = ' '
            first = self.vep[inc].flicker[0].pos
            best_neighbor = self.vep[inc + inc_q].flicker[0].pos

            # the textbox requires the pos to be centered on the middle of the box, so
            # we need to calculate the center of the vep stimulus boxes
            pos = (first[0] + best_neighbor[0]) / 2, (first[1] + best_neighbor[1]) / 2
            self.text_boxes += [visual.TextBox2(
                win=self.window,
                text=text,
                font=self.stimuli_font,
                units='height',
                pos=pos,
                color=color[0],
                colorSpace='rgb',
                size=[size, size],
                alignment='center',
                anchor='center',
                borderWidth=2,
                borderColor='white',
                letterHeight=self.stimuli_height / 2)]
            inc += self.vep_type
        return self.text_boxes

    def _reset_text_boxes(self) -> None:
        """Reset text boxes.

        This method resets the text boxes to the blank state.
        """
        for text_box in self.text_boxes:
            text_box.setText(' ')

    def _set_inquiry(self) -> List[visual.TextBox2]:
        """Generate Inquiry.

        This method sets the correct inquiry text for each text boxes.
        """
        # generate the inquiry stimuli
        assert len(
            self.stimuli_inquiry) == self.vep_type, (
                f"stmuli inquiry {len(self.stimuli_inquiry)} must be the same length as vep type {self.vep_type}")
        assert len(
            self.stimuli_colors) == self.vep_type, (
                f"stmuli colors {len(self.stimuli_colors)} must be the same length as vep type {self.vep_type}")
        for box_content, color, box in zip(self.stimuli_inquiry, self.stimuli_colors, self.text_boxes):
            text = ' '.join(box_content)
            box.text = text
            box.color = color[0]
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
