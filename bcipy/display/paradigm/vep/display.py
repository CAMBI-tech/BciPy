import logging
from tracemalloc import start
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
from bcipy.helpers.task import SPACE_CHAR, get_key_press
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger
from bcipy.helpers.stimuli import resize_image, get_fixation


def get_data_np(nbits=32, ncodes=4) -> List[List[int]]:
    res = []
    np.random.seed(1)
    for _ in range(ncodes):
        res.append(np.random.randint(2, size=nbits))
    return res


class VEPBox:

    def __init__(self, flicker: List[Tuple[visual.GratingStim, visual.GratingStim]], code: List[int]) -> None:
        self.flicker = flicker
        self.code = code

    def frame_on(self):
        self.flicker[0].draw()

    def frame_off(self):
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

        # self.pos = [(-.3, -.3), (.3, -.3), (.3, .3), (-.3, .3)]
        self.colors = [('white', 'black'), ('red', 'green'), ('blue', 'yellow'), ('orange', 'green')]
        if not codes:
            self.codes = get_data_np(nbits=self.refresh_rate, ncodes=4)
        else:
            self.codes = codes

        # build the VEP stimuli
        self.vep = []
        self.text_boxes = []
        self.vep_type = 4
        self.vep_box_size = .1

    def do_fixation(self):
        # draw fixation cross
        self.fixation.draw()
        self.draw_static()
        self.window.flip()
        core.wait(self.stimuli_timing[1])

    def do_inquiry(self) -> List[float]:
        """Do the inquiry."""

        timing = []
        if self.first_run:
            self._trigger_pulse()
            self._build_vep_corner_stimuli(self.stimuli_pos, self.codes, self.colors, self.vep_box_size)

        # inquiry = self._generate_inquiry() These will likely be text boxes at self.pos
    
        self.do_fixation()

        # timing += self.animate_inquiry()
        timing += self.stimulate()

        # end session!
        self.draw_static()
        self.window.flip()
        return timing

    def animate_inquiry(self, inquiry: List[List[str]]) -> List[float]:
        """Animate the inquiry.
        
        Inquiry is a list of lists of strings. Each list contains what stimuli to display for each box defined in self.vep.
        """
        self.trigger_callback.reset()
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'VEP_INQUIRY')
        self.static_clock.reset()
        
        return self.trigger_callback.timing

    def stimulate(self):
        """Stimulate.
        
        This is the main display function of the VEP paradigm. It is responsible for drawing the VEP flickering stimuli,
        to inquire as to whether the subject is interested in the stimuli, and to wait for a response.
        
        It assumes that the VEP stimuli are already constructed using the _build_vep_corner_stimuli function or similar.
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
    
    def draw_static(self):
        """Draw static elements in a stimulus."""
        self.task.draw()
        for info in self.info_text:
            info.draw()
    
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
        self.draw_static()
        self.window.flip()

    def update_task(self, text: str, color: str, pos: Optional[Tuple] = None) -> None:
        self.task.text = text
        self.task.color = color
        if pos:
            self.task.pos = pos

    def _build_vep_corner_stimuli(self, pos, codes, colors, box_size) -> List[visual.GratingStim]:
        """Build the corner stimuli for the VEP.

        Args:
            window: psychopy window object
        """
        self.logger.info('building VEP stimuli')
        assert len(codes[0]) >= self.refresh_rate, 'Code must be longer than refresh rate'
        corner_stim = []
        for pos, code, color in zip(pos, codes, colors):
            corner_stim += self._build_2x2_vep_grid(pos, box_size, color, code)

        return corner_stim
        

    def _build_1x1_vep_grid(self, position, size, color, code):
        assert len(code) >= self.refresh_rate, 'Code must be longer than refresh rate'
        pattern1 = visual.GratingStim(win=self.window, name=f'1x1-1-{position}',units='height', 
                            tex=None, mask=None,
                            ori=0, pos=position, size=size, sf=1, phase=0.0,
                            color=color[0], colorSpace='rgb', opacity=1, 
                            texRes=256, interpolate=True, depth=-1.0)
        pattern2 = visual.GratingStim(win=self.window, name=f'1x1-2-{position}',units='height', 
                        tex=None, mask=None,
                        ori=0, pos=position, size=size, sf=1, phase=0,
                        color=color[1], colorSpace='rgb', opacity=1,
                        texRes=256, interpolate=True, depth=-2.0)
        box = [VEPBox(flicker=[pattern1, pattern2], code=code)]
        self.vep += box
        return box
    
    def _build_2x2_vep_grid(self, position, size, color, code):

        assert len(code) >= self.refresh_rate, 'Code must be longer than refresh rate'
        starting_position = position
        positions = [
            starting_position,
            [starting_position[0] + size, starting_position[1]],
            [starting_position[0] + size, starting_position[1] + size],
            [starting_position[0], starting_position[1] + size],
        ]
        box = []
        for pos in positions:
            pattern1 = visual.GratingStim(win=self.window, name=f'2x2-1-{pos}',units='height', 
                            tex=None, mask=None,
                            ori=0, pos=pos, size=size, sf=1, phase=0.0,
                            color=color[0], colorSpace='rgb', opacity=1, 
                            texRes=256, interpolate=True, depth=-1.0)
            pattern2 = visual.GratingStim(win=self.window, name=f'2x2-2-{pos}',units='height', 
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

    def _generate_inquiry(self) -> None:
        """Generate Inquiry.

        This method generates the inquiry stimuli for the experiment.
        """
        # generate the inquiry stimuli
        assert len(self.stimuli_inquiry) == self.vep_type, "stmuli inquiry must be the same length as vep"
        assert len(self.stimuli_colors) == self.vep_type, "stmuli colors  must be the same length as vep"
        self.text_boxes = []
        size = self.vep_box_size * self.vep_type
        for box_content, pos, color in zip(self.stimuli_inquiry, self.stimuli_pos, self.stimuli_colors):
            self.text_boxes += visual.TextBox2(
                win=self.window, text=box_content[0], font=self.stimuli_font, pos=pos, color=color,
                size=[size, size], alignment='center', borderWidth=1, borderColor='white',
                letterHeight=self.stimuli_height)
        return self.text_boxes