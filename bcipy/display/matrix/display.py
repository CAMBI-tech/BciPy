from typing import List, Optional
import logging

from psychopy import visual, core

from bcipy.acquisition.marker_writer import NullMarkerWriter, MarkerWriter
from bcipy.display import Display, StimuliProperties, TaskDisplayProperties, InformationProperties, BCIPY_LOGO_PATH
from bcipy.helpers.task import SPACE_CHAR
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger
from bcipy.helpers.task import alphabet


class MatrixDisplay(Display):

    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: core.Clock,
            stimuli: StimuliProperties,
            task_display: TaskDisplayProperties,
            info: InformationProperties,
            # window.callOnFlip(callback()) --> writes a marker to LSL
            marker_writer: Optional[MarkerWriter] = NullMarkerWriter(),
            trigger_type: str = 'text',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False,
            symbol_set=None):
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
        # TODO: error on non-text stimuli
        # assert self.is_txt_stim == 'text'
        self.stim_length = stimuli.stim_length

        self.full_screen = full_screen

        self.symbol_set = symbol_set or alphabet()

        self.staticPeriod = static_clock

        self.position = stimuli.stim_pos
        self.position_increment = 0.2
        self.max_grid_width = 0.7
        self.stim_registry = {}
        self.opacity = 0.2

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

        self.scp = True  # for future row / col integration

        # Create multiple text objects based on input FIX THIS!
        # self.info = info
        # self.text = info.build_info_text(window)
        #  Letter selected

        # Create initial stimuli object for updating
        self.sti = stimuli.build_init_stimuli(window)

    def schedule_to(self, stimuli: list, timing: list, colors: list) -> None:
        self.stimuli_inquiry = stimuli
        self.stimuli_timing = timing
        self.stimuli_colors = colors

    def do_inquiry(self) -> List[float]:
        """Do inquiry.

        Animates an inquiry of stimuli and returns a list of stimuli trigger timing.
        """
        if self.scp:
            self.animate_scp()

    def build_grid(self) -> None:

        pos = self.position
        for sym in self.symbol_set:
            text_stim = visual.TextStim(
                self.window,
                text=sym,
                opacity=self.opacity,
                pos=pos,
                height=self.stimuli_height)
            self.stim_registry[sym] = text_stim
            text_stim.draw()

            x, y = pos
            x += self.position_increment
            if x >= self.max_grid_width:
                y -= self.position_increment
                x = self.position[0]
            pos = (x, y)

    def animate_scp(self) -> None:
        i = 0
        for sym in self.stimuli_inquiry:
            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                sym)
            self.window.callOnFlip(self.marker_writer.push_marker, sym)
            self.build_grid()
            self.stim_registry[sym].opacity = 1
            self.stim_registry[sym].draw()
            self.window.flip()
            core.wait(self.stimuli_timing[i])
            self.stim_registry[sym].opacity = 0.0
            self.stim_registry[sym].draw()
            i += 1

    def wait_screen(self) -> None:
        """Wait Screen.

        Define what happens on the screen when a user pauses a session.
        """
        #
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

        except Exception:
            self.logger.exception(f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            pass

        # Draw and flip the screen.
        wait_message.draw()
        self.window.flip()

    def update_task(self) -> None:
        """Update Task.

        Update any task related display items not related to the inquiry. Ex. stimuli count 1/200.
        """
        pass


if __name__ == '__main__':

    display = MatrixDisplay()
