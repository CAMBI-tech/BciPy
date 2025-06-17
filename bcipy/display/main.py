"""Main display module.

This module provides the core display functionality for BciPy, including base classes
and utilities for creating and managing visual stimuli in BCI paradigms.
"""

# mypy: disable-error-code="assignment,empty-body"
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, NamedTuple, Optional, Tuple, Type, Union, Dict

from psychopy import visual

from bcipy.display.components.button_press_handler import (
    AcceptButtonPressHandler, ButtonPressHandler,
    PreviewOnlyButtonPressHandler, RejectButtonPressHandler)
from bcipy.helpers.clock import Clock
from bcipy.helpers.utils import get_screen_info


class Display(ABC):
    """Base class for BciPy displays.

    This abstract class defines the core interface and functionality necessary for
    task executions that require a display. It provides methods for stimulus
    presentation, timing control, and task management.

    Attributes:
        window (visual.Window): PsychoPy window for display.
        timing_clock (Clock): Clock for timing control.
        experiment_clock (Clock): Clock for experiment timing.
        stimuli_inquiry (List[str]): List of stimuli to present.
        stimuli_colors (List[str]): List of colors for each stimulus.
        stimuli_timing (List[float]): List of presentation durations.
        task (Any): Task-related information.
        info_text (List[Any]): Information text to display.
        first_stim_time (float): Time of first stimulus presentation.
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
        """Perform an inquiry of stimuli.

        Animates an inquiry of stimuli and returns a list of stimuli trigger timing.

        Returns:
            List[Tuple[str, float]]: List of (stimulus, timing) pairs.
        """
        ...

    @abstractmethod
    def wait_screen(self, *args: Any, **kwargs: Any) -> None:
        """Display a wait screen.

        Define what happens on the screen when a user pauses a session.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ...

    @abstractmethod
    def update_task_bar(self, *args: Any, **kwargs: Any) -> None:
        """Update task bar display.

        Update any taskbar-related display items not related to the inquiry.
        Example: stimuli count 1/200.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ...

    def schedule_to(self, stimuli: List[str], timing: List[float], colors: List[str]) -> None:
        """Schedule stimuli elements.

        Schedule stimuli elements (works as a buffer) before calling do_inquiry.

        Args:
            stimuli (List[str]): List of stimuli to present.
            timing (List[float]): List of presentation durations.
            colors (List[str]): List of colors for each stimulus.
        """
        ...

    def draw_static(self) -> None:
        """Draw static elements.

        Displays task information not related to the inquiry.
        """
        ...

    def preview_inquiry(self, *args: Any, **kwargs: Any) -> List[float]:
        """Preview an inquiry before presentation.

        Display an inquiry or instruction beforehand to the user. This should be called
        before do_inquiry. This can be used to determine if the desired stimuli is present
        before displaying them more laboriously or prompting users before the inquiry.

        Note:
            All stimuli elements (stimuli, timing, colors) must be set on the display
            before calling this method. This implies something like schedule_to is called.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[float]: List of timing information for the preview.
        """
        ...


def init_display_window(parameters: Dict[str, Any]) -> visual.Window:
    """Initialize the main display window.

    Function to initialize main display window needed for all later stimuli presentation.

    See Psychopy official documentation for more information and working demos:
        http://www.psychopy.org/api/visual/window.html

    Args:
        parameters (Dict[str, Any]): Dictionary containing window configuration parameters.

    Returns:
        visual.Window: Initialized PsychoPy window for display.
    """
    # Check if full_screen mode is set and get necessary values
    if parameters['full_screen']:
        # set window attributes based on resolution
        screen_info = get_screen_info()
        window_height = screen_info.height
        window_width = screen_info.width
        full_screen = True
    else:
        # set window attributes directly from parameters file
        window_height = parameters['window_height']
        window_width = parameters['window_width']
        full_screen = False

    # Initialize PsychoPy Window for Main Display of Stimuli
    display_window = visual.Window(
        size=[window_width, window_height],
        screen=parameters['stim_screen'],
        allowGUI=False,
        useFBO=False,
        fullscr=full_screen,
        allowStencil=False,
        monitor='mainMonitor',
        winType='pyglet',
        units='norm',
        waitBlanking=False,
        color=parameters['background_color'])

    return display_window


class StimuliProperties:
    """Encapsulation of properties for core stimuli presentation.

    This class manages the properties and configuration for presenting stimuli
    in a paradigm, including text and image-based stimuli.

    Attributes:
        stim_font (str): Font to use for text stimuli.
        stim_pos (Union[Tuple[float, float], List[Tuple[float, float]]]): Position(s) for stimuli.
        stim_height (float): Height of stimuli.
        stim_inquiry (List[str]): List of stimuli to present.
        stim_colors (List[str]): List of colors for each stimulus.
        stim_timing (List[float]): List of presentation durations.
        is_txt_stim (bool): Whether stimuli are text-based.
        stim_length (int): Number of stimuli.
        sti (Optional[Union[visual.TextStim, visual.ImageStim]]): Stimulus object.
        prompt_time (Optional[float]): Time to display target prompt.
        layout (Optional[str]): Layout of stimuli (e.g., 'ALPHABET' or 'QWERTY').
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
            layout: Optional[str] = None) -> None:
        """Initialize Stimuli Properties.

        Args:
            stim_font (str): Font to use for text stimuli.
            stim_pos (Union[Tuple[float, float], List[Tuple[float, float]]]): Position(s) for stimuli.
            stim_height (float): Height of stimuli.
            stim_inquiry (Optional[List[str]]): List of stimuli to present. Defaults to None.
            stim_colors (Optional[List[str]]): List of colors for each stimulus. Defaults to None.
            stim_timing (Optional[List[float]]): List of presentation durations. Defaults to None.
            is_txt_stim (bool): Whether stimuli are text-based. Defaults to True.
            prompt_time (Optional[float]): Time to display target prompt. Defaults to None.
            layout (Optional[str]): Layout of stimuli. Defaults to None.
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
        """Build initial stimulus object.

        This method constructs the stimuli object which can be updated later. This is more
        performant than creating a new stimuli each call. It can create either an image or
        text stimuli based on the boolean self.is_txt_stim.

        Args:
            window (visual.Window): PsychoPy window for display.

        Returns:
            Union[visual.TextStim, visual.ImageStim]: The created stimulus object.
        """
        if self.is_txt_stim:
            self.sti = visual.TextStim(
                win=window,
                color='white',
                height=self.stim_height,
                text='',
                font=self.stim_font,
                pos=self.stim_pos,
                wrapWidth=None,
                colorSpace='rgb',
                opacity=1,
                depth=-6.0)
        else:
            self.sti = visual.ImageStim(
                win=window,
                image=None,
                mask=None,
                pos=self.stim_pos,
                ori=0.0)
        return self.sti


class InformationProperties:
    """Encapsulation of properties for task information presentation.

    This class manages the properties and configuration for displaying task-related
    information, feedback, and static text in an RSVP paradigm.

    Attributes:
        info_color (List[str]): List of colors for information text.
        info_text (List[str]): List of information text to display.
        info_font (List[str]): List of fonts for information text.
        info_pos (List[Tuple[float, float]]): List of positions for information text.
        info_height (List[float]): List of heights for information text.
        text_stim (List[visual.TextStim]): List of text stimulus objects.
    """

    def __init__(
            self,
            info_color: List[str],
            info_text: List[str],
            info_font: List[str],
            info_pos: List[Tuple[float, float]],
            info_height: List[float]) -> None:
        """Initialize Information Properties.

        Args:
            info_color (List[str]): List of colors for information text.
            info_text (List[str]): List of information text to display.
            info_font (List[str]): List of fonts for information text.
            info_pos (List[Tuple[float, float]]): List of positions for information text.
            info_height (List[float]): List of heights for information text.
        """
        self.info_color = info_color
        self.info_text = info_text
        self.info_font = info_font
        self.info_pos = info_pos
        self.info_height = info_height

    def build_info_text(self, window: visual.Window) -> List[visual.TextStim]:
        """Build information text stimuli.

        Constructs a list of Information stimuli to display.

        Args:
            window (visual.Window): PsychoPy window for display.

        Returns:
            List[visual.TextStim]: List of text stimulus objects.
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
                wrapWidth=None,
                colorSpace='rgb',
                opacity=1,
                depth=-6.0))
        return self.text_stim


class ButtonPressMode(Enum):
    """Represents the possible meanings for a button press.

    Used when implementing Inquiry Preview functionality to determine the
    action to take based on user input.
    """
    NOTHING = 0
    ACCEPT = 1
    REJECT = 2


class PreviewParams(NamedTuple):
    """Parameters for Inquiry Preview functionality.

    This class defines the configuration parameters needed for the Inquiry Preview
    feature, which allows users to preview stimuli before presentation.

    Attributes:
        show_preview_inquiry (bool): Whether to show preview.
        preview_inquiry_length (float): Duration of preview.
        preview_inquiry_key_input (str): Key to use for preview input.
        preview_inquiry_progress_method (int): Method for handling preview progress.
        preview_inquiry_isi (float): Inter-stimulus interval for preview.
        preview_box_text_size (float): Text size for preview box.
    """

    show_preview_inquiry: bool
    preview_inquiry_length: float
    preview_inquiry_key_input: str
    preview_inquiry_progress_method: int
    preview_inquiry_isi: float
    preview_box_text_size: float

    @property
    def button_press_mode(self) -> ButtonPressMode:
        """Get the button press mode from the progress method.

        Returns:
            ButtonPressMode: The mode indicated by the progress method.
        """
        return ButtonPressMode(self.preview_inquiry_progress_method)


def get_button_handler_class(
        mode: ButtonPressMode) -> Type[ButtonPressHandler]:
    """Get the appropriate button handler class for the given mode.

    Args:
        mode (ButtonPressMode): The button press mode to handle.

    Returns:
        Type[ButtonPressHandler]: The appropriate handler class.
    """
    mapping = {
        ButtonPressMode.NOTHING: PreviewOnlyButtonPressHandler,
        ButtonPressMode.ACCEPT: AcceptButtonPressHandler,
        ButtonPressMode.REJECT: RejectButtonPressHandler
    }
    return mapping[mode]


def init_preview_button_handler(params: PreviewParams,
                                experiment_clock: Clock) -> ButtonPressHandler:
    """Initialize a button press handler for inquiry preview.

    Args:
        params (PreviewParams): Preview configuration parameters.
        experiment_clock (Clock): Clock for experiment timing.

    Returns:
        ButtonPressHandler: Configured button press handler.
    """
    make_handler = get_button_handler_class(params.button_press_mode)
    return make_handler(max_wait=params.preview_inquiry_length,
                        key_input=params.preview_inquiry_key_input,
                        clock=experiment_clock)


class VEPStimuliProperties(StimuliProperties):
    """Properties for VEP (Visual Evoked Potential) stimuli.

    This class extends StimuliProperties to provide specific functionality
    for VEP-based paradigms.

    Attributes:
        animation_seconds (float): Duration of animation.
    """

    def __init__(self,
                 stim_font: str,
                 stim_pos: List[Tuple[float, float]],
                 stim_height: float,
                 timing: List[float],
                 stim_color: List[str],
                 inquiry: List[List[Any]],
                 stim_length: int = 1,
                 animation_seconds: float = 1.0) -> None:
        """Initialize VEP Stimuli Properties.

        Args:
            stim_font (str): Font to use for text stimuli.
            stim_pos (List[Tuple[float, float]]): Positions for stimuli.
            stim_height (float): Height of stimuli.
            timing (List[float]): List of presentation durations.
            stim_color (List[str]): List of colors for each stimulus.
            inquiry (List[List[Any]]): List of inquiry stimuli.
            stim_length (int, optional): Number of stimuli. Defaults to 1.
            animation_seconds (float, optional): Duration of animation. Defaults to 1.0.
        """
        # static properties
        self.stim_font = stim_font
        self.stim_height = stim_height
        self.is_txt_stim = True
        self.stim_length = stim_length  # how many times to flicker
        self.stim_pos = stim_pos

        # dynamic property. List of length 3. 1. prompt; 2. fixation; 3. inquiry
        self.stim_timing = timing

        # dynamic properties, must be a list of lists where each list is a different box
        self.stim_colors = stim_color
        self.stim_inquiry = inquiry
        self.animation_seconds = animation_seconds

    def build_init_stimuli(self, window: visual.Window) -> None:
        """Build initial VEP stimuli.

        Args:
            window (visual.Window): PsychoPy window for display.
        """
        ...
