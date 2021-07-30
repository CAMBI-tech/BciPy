from typing import List, Optional
import logging

from psychopy import visual, core

from bcipy.acquisition.marker_writer import NullMarkerWriter, MarkerWriter
from bcipy.display import Display, StimuliProperties, TaskDisplayProperties, InformationProperties 
from bcipy.helpers.task import SPACE_CHAR
from bcipy.helpers.triggers import TriggerCallback, _calibration_trigger


class MatrixDisplay(Display):
     
    def __init__(
            self,
            window: visual.Window,
            static_clock,
            experiment_clock: core.Clock,
            stimuli: StimuliProperties, # refactor (lives in RSVP; useful!)
            task_display: TaskDisplayProperties, # refactor (lives in RSVP; useful!)
            info: InformationProperties, # refactor (lives in RSVP; useful!)
            marker_writer: Optional[MarkerWriter] = NullMarkerWriter(), # window.callOnFlip(callback()) --> writes a marker to LSL 
            trigger_type: str = 'text',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False):
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

    def do_inquiry(self) -> List[float]:
        """Do inquiry.

        Animates an inquiry of stimuli and returns a list of stimuli trigger timing.
        """

    def build_grid(self):
        """visual.TextBox2(
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
        """ 

        # We want to initialize all symbols (alpabets)

        # stim = visual.TextStim(win=self.win,
                    #                        height=self.stim_height,
                    #                        text='A',
                    #                        font=self.stimuli_font, pos=pos,
                    #                        wrapWidth=None, colorSpace='rgb',
                    #                        opacity=1, depth=-6.0)
        # stim.draw()
        # window.flip()
        # [1]. Use index and list: self.stimuli.append(stim) *** WHAT DATA STRUCTURE TO STORE THIS IS IN ORDER TO FLASH LATER!
        # [2]. Create a dictionary using the symbol as key and stimuli as value.
        # self.stimuli['B'] = stimuli # setting 

    def animate_scp():
        # self.stimuli['B'].color = 'white' # modifying later
        # self.stimuli['B'].draw()
        # self.window.flip()
        pass

    def wait_screen(self) -> None:
        """Wait Screen.

        Define what happens on the screen when a user pauses a session.
        """
        pass

    def update_task(self) -> None:
        """Update Task.

        Update any task related display items not related to the inquiry. Ex. stimuli count 1/200.
        """
        pass

if __name__ == '__main__':

    display = MatrixDisplay()