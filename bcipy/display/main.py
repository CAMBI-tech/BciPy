from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Tuple, Union

from psychopy import visual

from bcipy.helpers.clock import Clock
from bcipy.helpers.system_utils import get_screen_resolution


class Display(ABC):
    """Display.

    Base class for BciPy displays. This defines the logic necessary for task executions that require a display.
    """

    window: visual.Window = None
    timing_clock: Clock = None
    experiment_clock: Clock = None
    logger: Logger = None
    stimuli_inquiry: List[str] = None
    stimuli_colors: List[str] = None
    stimuli_timing: List[float] = None
    task = None

    @abstractmethod
    def do_inquiry(self) -> List[float]:
        """Do inquiry.

        Animates an inquiry of stimuli and returns a list of stimuli trigger timing.
        """
        ...

    @abstractmethod
    def wait_screen(self) -> None:
        """Wait Screen.

        Define what happens on the screen when a user pauses a session.
        """
        ...

    @abstractmethod
    def update_task(self) -> None:
        """Update Task.

        Update any task related display items not related to the inquiry. Ex. stimuli count 1/200.
        """
        ...

    def schedule_to(self, stimuli: list, timing: list, colors: list) -> None:
        """Schedule To.

        Schedule stimuli elements (works as a buffer) before calling do_inquiry.
        """
        ...

    def draw_static(self) -> None:
        """Draw Static.

        Displays task information not related to the inquiry.
        """
        ...

    def preview_inquiry(self) -> List[float]:
        """Preview Inquiry.

        Display an inquiry or instruction beforehand to the user. This should be called before do_inquiry.
        This can be used to determine if the desired stimuli is present before displaying them more laboriusly
        or prompting users before the inquiry.
        All stimuli elements (stimuli, timing, colors) must be set on the display before calling this method.
        This implies, something like schedule_to is called.
        """
        ...


def init_display_window(parameters):
    """
    Init Display Window.

    Function to Initialize main display window
        needed for all later stimuli presentation.

    See Psychopy official documentation for more information and working demos:
        http://www.psychopy.org/api/visual/window.html
    """

    # Check is full_screen mode is set and get necessary values
    if parameters['full_screen']:

        # set window attributes based on resolution
        window_width, window_height = get_screen_resolution()

        # set full screen mode to true (removes os dock, explorer etc.)
        full_screen = True

    # otherwise, get user defined window attributes
    else:

        # set window attributes directly from parameters file
        window_height = parameters['window_height']
        window_width = parameters['window_width']

        # make sure full screen is set to false
        full_screen = False

    # Initialize PsychoPy Window for Main Display of Stimuli
    display_window = visual.Window(
        size=[window_width,
              window_height],
        screen=parameters['stim_screen'],
        allowGUI=False,
        useFBO=False,
        fullscr=full_screen,
        allowStencil=False,
        monitor='mainMonitor',
        winType='pyglet', units='norm', waitBlanking=False,
        color=parameters['background_color'])

    # Return display window to caller
    return display_window


class StimuliProperties:
    """"Stimuli Properties.

    An encapsulation of properties relevant to core stimuli presentation in an RSVP or Matrix paradigm.
    """

    def __init__(
            self,
            stim_font: str,
            stim_pos: Tuple[float, float],
            stim_height: float,
            stim_inquiry: List[str],
            stim_colors: List[str],
            stim_timing: List[float],
            is_txt_stim: bool,
            prompt_time: float = None):
        """Initialize Stimuli Parameters.

        stim_font(List[str]): Ordered list of colors to apply to information stimuli
        stim_pos(Tuple[float, float]): Position on window where the stimuli will be presented
        stim_height(float): Height of all stimuli
        stim_inquiry(List[str]): Ordered list of text to build stimuli with
        stim_colors(List[str]): Ordered list of colors to apply to stimuli
        stim_timing(List[float]): Ordered list of timing to apply to an inquiry using the stimuli
        is_txt_stim(bool): Whether or not this is a text based stimuli (False implies image based)
        prompt_time(float): Time to display target prompt for at the beggining of inquiry
        """
        self.stim_font = stim_font
        self.stim_pos = stim_pos
        self.stim_height = stim_height
        self.stim_inquiry = stim_inquiry
        self.stim_colors = stim_colors
        self.stim_timing = stim_timing
        self.is_txt_stim = is_txt_stim
        self.stim_length = len(self.stim_inquiry)
        self.sti = None
        self.prompt_time = prompt_time

    def build_init_stimuli(self, window: visual.Window) -> Union[visual.TextStim, visual.ImageStim]:
        """"Build Initial Stimuli.

        This method constructs the stimuli object which can be updated later. This is more
            performant than creating a new stimuli each call. It can create either an image or text stimuli
            based on the boolean self.is_txt_stim.
        """
        if self.is_txt_stim:
            self.sti = visual.TextStim(
                win=window,
                color='white',
                height=self.stim_height,
                text='',
                font=self.stim_font,
                pos=self.stim_pos,
                wrapWidth=None, colorSpace='rgb',
                opacity=1, depth=-6.0)
        else:
            self.sti = visual.ImageStim(
                win=window,
                image=None,
                mask=None,
                pos=self.stim_pos,
                ori=0.0)
        return self.sti


class InformationProperties:
    """"Information Properties.

    An encapsulation of properties relevant to task information presentation in an RSVP paradigm. This could be
        messaging relevant to feedback or static text to remain on screen not related to task tracking.
    """

    def __init__(
            self,
            info_color: List[str],
            info_text: List[str],
            info_font: List[str],
            info_pos: List[Tuple[float, float]],
            info_height: List[float]):
        """Initialize Information Parameters.

        info_color(List[str]): Ordered list of colors to apply to information stimuli
        info_text(List[str]): Ordered list of text to apply to information stimuli
        info_font(List[str]): Ordered list of font to apply to information stimuli
        info_pos(Tuple[float, float]): Position on window where the Information stimuli will be presented
        info_height(List[float]): Ordered list of height of Information stimuli
        """
        self.info_color = info_color
        self.info_text = info_text
        self.info_font = info_font
        self.info_pos = info_pos
        self.info_height = info_height

    def build_info_text(self, window: visual.Window) -> List[visual.TextStim]:
        """"Build Information Text.

        Constructs a list of Information stimuli to display.
        """
        self.text_stim = []
        for idx in range(len(self.info_text)):
            self.text_stim.append(visual.TextStim(
                win=window,
                color=self.info_color[idx],
                height=self.info_height[idx],
                text=self.info_text[idx],
                font=self.info_font[idx],
                pos=self.info_pos[idx],
                wrapWidth=None, colorSpace='rgb',
                opacity=1, depth=-6.0))
        return self.text_stim


class TaskDisplayProperties:
    """"Task Dispay Properties.

    An encapsulation of properties relevant to task stimuli presentation in an RSVP paradigm.
    """

    def __init__(
            self,
            task_color: List[str],
            task_font: str,
            task_pos: Tuple[float, float],
            task_height: float,
            task_text: str):
        """Initialize Task Display Parameters.

        task_color(List[str]): Ordered list of colors to apply to task stimuli
        task_font(str): Font to apply to all task stimuli
        task_pos(Tuple[float, float]): Position on the screen where to present to task text
        task_height(float): Height of all task text stimuli
        task_text(str): Task text to apply to stimuli
        """
        self.task_color = task_color
        self.task_font = task_font
        self.task_pos = task_pos
        self.task_height = task_height
        self.task_text = task_text
        self.task = None

    def build_task(self, window: visual.Window) -> visual.TextStim:
        """"Build Task.

        This method constructs the task stimuli object which can be updated later. This is more
            performant than creating a new stimuli for each update in task state.
        """
        self.task = visual.TextStim(
            win=window,
            color=self.task_color[0],
            height=self.task_height,
            text=self.task_text,
            font=self.task_font,
            pos=self.task_pos,
            wrapWidth=None, colorSpace='rgb',
            opacity=1, depth=-6.0)
        return self.task


class PreviewInquiryProperties:
    """"Preview Inquiry Properties.
    An encapsulation of properties relevant to preview_inquiry() operation.
    """

    def __init__(
            self,
            preview_only: bool,
            preview_inquiry_length: float,
            preview_inquiry_progress_method: int,
            preview_inquiry_key_input: str,
            preview_inquiry_isi: float):
        """Initialize Inquiry Preview Parameters.
        preview_inquiry_length(float): Length of time in seconds to present the inquiry preview
        preview_inquiry_progress_method(int): Method of progression for inquiry preview.
            0 == preview only; 1 == press to accept inquiry; 2 == press to skip inquiry.
        preview_inquiry_key_input(str): Defines which key should be listened to for progressing
        preview_inquiry_isi(float): Length of time after displaying the inquiry preview to display a blank screen
        """
        self.preview_inquiry_length = preview_inquiry_length
        self.preview_inquiry_key_input = preview_inquiry_key_input
        self.press_to_accept = True if preview_inquiry_progress_method == 1 else False
        self.preview_only = preview_only
        self.preview_inquiry_isi = preview_inquiry_isi


class VEPStimuliProperties(StimuliProperties):

    def __init__(
            self,
            stim_font: str,
            stim_pos: List[Tuple[float, float]],
            stim_height: float,
            timing: Tuple[float, float, float] = None,
            stim_color: List[List[str]] = None,
            inquiry: List[List[str]] = None,
            stim_length: int = 1):
        """Initialize VEP Stimuli Parameters.
        stim_color(List[str]): Ordered list of colors to apply to VEP stimuli
        stim_font(str): Font to apply to all VEP stimuli
        stim_pos(List[Tuple[float, float]]): Position on the screen where to present to VEP text
        stim_height(float): Height of all VEP text stimuli
        """
        # static properties
        self.stim_font = stim_font
        self.stim_height = stim_height
        self.is_txt_stim = True
        self.stim_length = stim_length  # how many times to flicker
        self.stim_pos = stim_pos

        # dynamic property. List of length 3. 1. prompt; 2. fixation; 3. inquiry
        self.stim_timing = timing

        # dynamic properties, must be a a list of lists where each list is a different box
        self.stim_colors = stim_color
        self.stim_inquiry = inquiry

    def build_init_stimuli(self, window: visual.Window) -> None:
        """"Build Initial Stimuli."""
        ...
