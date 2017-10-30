from __future__ import division
from psychopy import visual


def init_display_window(parameters):
    """
    Init Display Window.

    Function to Initialize main display window
        needed for all later stimuli presentation.
    """

    # Initialize PsychoPy Window for Main Display of Stimuli
    display_window = visual.Window(
        size=[500, 500],
        fullscr=False, screen=0,
        allowGUI=False, allowStencil=False, monitor='mainMonitor',
        color='black', colorSpace='rgb', blendMode='avg', waitBlanking=True)

    # Return it to caller
    return display_window
