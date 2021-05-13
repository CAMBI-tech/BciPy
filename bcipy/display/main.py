from abc import ABC, abstractmethod
from logging import Logger
from psychopy import visual, core
from typing import List, Optional

from bcipy.acquisition.marker_writer import MarkerWriter
from bcipy.helpers.system_utils import get_screen_resolution


class Display(ABC):
    """Display.

    Base class for BciPy displays. This defines the logic necessary for task excecutions that require a display.
    """

    window: visual.Window = None
    timing_clock: core.Clock = None
    experiment_clock: core.Clock = None
    marker_writer: Optional[MarkerWriter] = None
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

        Update any task related display items not releated to the inquiry. Ex. stimuli count 1/200.
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
