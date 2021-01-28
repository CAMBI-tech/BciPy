from psychopy import visual
from bcipy.helpers.system_utils import get_screen_resolution


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
        screen=int(parameters['stim_screen']),
        allowGUI=False,
        useFBO=False,
        fullscr=full_screen,
        allowStencil=False,
        monitor='mainMonitor',
        winType='pyglet', units='norm', waitBlanking=False,
        color=parameters['background_color'])

    # Return display window to caller
    return display_window
