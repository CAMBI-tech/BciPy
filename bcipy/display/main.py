# mypy: disable-error-code="assignment,empty-body"
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, NamedTuple, Optional, Tuple, Type, Union

from psychopy import visual

from bcipy.display.components.button_press_handler import (
    AcceptButtonPressHandler, ButtonPressHandler,
    PreviewOnlyButtonPressHandler, RejectButtonPressHandler)
from bcipy.helpers.clock import Clock
from bcipy.helpers.system_utils import get_screen_info


class Display(ABC):
    """Display.

    Base class for BciPy displays. This defines the logic necessary for task executions that require a display.
    """

    window: visual.Window = None
    timing_clock: Clock = None
    experiment_clock: Clock = None
    stimuli_inquiry: List[str] = None
    stimuli_colors: List[str] = None
    stimuli_timing: List[float] = None
    task = None
    info_text: List[Any] = None
    first_stim_time: float = None

    @abstractmethod
    def do_inquiry(self) -> List[Tuple[str, float]]:
        """Do inquiry.

        Animates an inquiry of stimuli and returns a list of stimuli trigger timing.
        """
        ...

    @abstractmethod
    def wait_screen(self, *args, **kwargs) -> None:
        """Wait Screen.

        Define what happens on the screen when a user pauses a session.
        """
        ...

    @abstractmethod
    def update_task_bar(self, *args, **kwargs) -> None:
        """Update Task.

        Update any taskbar-related display items not related to the inquiry. Ex. stimuli count 1/200.
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

    def preview_inquiry(self, *args, **kwargs) -> List[float]:
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
        screen_info = get_screen_info()
        window_height = screen_info.height
        window_width = screen_info.width

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

    An encapsulation of properties relevant to core stimuli presentation in a paradigm.
    """

    def __init__(
            self,
            stim_font: str,
            stim_pos: Union[Tuple[float, float], List[Tuple[float, float]]],
            stim_height: float,
            stim_inquiry: Optional[List[str]] = None,
            stim_colors: Optional[List[str]] = None,
            stim_timing: Optional[List[float]] = None,
            is_txt_stim: bool = True,
            prompt_time: Optional[float] = None,
            layout: Optional[str] = None):
        """Initialize Stimuli Parameters.

        stim_font(List[str]): Ordered list of colors to apply to information stimuli
        stim_pos(Tuple[float, float]): Position on window where the stimuli will be presented
            or a list of positions (ex. for matrix displays)
        stim_height(float): Height of all stimuli
        stim_inquiry(List[str]): Ordered list of text to build stimuli with
        stim_colors(List[str]): Ordered list of colors to apply to stimuli
        stim_timing(List[float]): Ordered list of timing to apply to an inquiry using the stimuli
        is_txt_stim(bool): Whether or not this is a text based stimuli (False implies image based)
        prompt_time(float): Time to display target prompt for at the beginning of inquiry
        layout(str): Layout of stimuli on the screen (ex. 'ALPHABET' or 'QWERTY').
            This is only used for matrix displays.
        """
        self.stim_font = stim_font
        self.stim_pos = stim_pos
        self.stim_height = stim_height
        self.stim_inquiry = stim_inquiry or []
        self.stim_colors = stim_colors or []
        self.stim_timing = stim_timing or []
        self.is_txt_stim = is_txt_stim
        self.stim_length = len(self.stim_inquiry)
        self.sti = None
        self.prompt_time = prompt_time
        self.layout = layout

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


class ButtonPressMode(Enum):
    """Represents the possible meanings for a button press (when using an Inquiry Preview.)"""
    NOTHING = 0
    ACCEPT = 1
    REJECT = 2


class PreviewParams(NamedTuple):
    """Parameters relevant for the Inquiry Preview functionality.

    Create from an existing Parameters instance using:
    >>> parameters.instantiate(PreviewParams)
    """
    show_preview_inquiry: bool
    preview_inquiry_length: float
    preview_inquiry_key_input: str
    preview_inquiry_progress_method: int
    preview_inquiry_isi: float
    preview_box_text_size: float

    @property
    def button_press_mode(self):
        """Mode indicated by the inquiry progress method."""
        return ButtonPressMode(self.preview_inquiry_progress_method)


def get_button_handler_class(
        mode: ButtonPressMode) -> Type[ButtonPressHandler]:
    """Get the appropriate handler constructor for the given button press mode."""
    mapping = {
        ButtonPressMode.NOTHING: PreviewOnlyButtonPressHandler,
        ButtonPressMode.ACCEPT: AcceptButtonPressHandler,
        ButtonPressMode.REJECT: RejectButtonPressHandler
    }
    return mapping[mode]


def init_preview_button_handler(params: PreviewParams,
                                experiment_clock: Clock) -> ButtonPressHandler:
    """"Returns a button press handler for inquiry preview."""
    make_handler = get_button_handler_class(params.button_press_mode)
    return make_handler(max_wait=params.preview_inquiry_length,
                        key_input=params.preview_inquiry_key_input,
                        clock=experiment_clock)


class VEPStimuliProperties(StimuliProperties):

    def __init__(self,
                 stim_font: str,
                 stim_pos: List[Tuple[float, float]],
                 stim_height: float,
                 timing: List[float],
                 stim_color: List[str],
                 inquiry: List[List[Any]],
                 stim_length: int = 1,
                 animation_seconds: float = 1.0):
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
        self.animation_seconds = animation_seconds

    def build_init_stimuli(self, window: visual.Window) -> None:
        """"Build Initial Stimuli."""
        ...
