# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import visual
from utils.get_system_info import get_system_info


def init_display_window(parameters):
    """
    Init Display Window.

    Function to Initialize main display window
        needed for all later stimuli presentation.

    See Psychopy official documentation for more information and working demos:
        http://www.psychopy.org/api/visual/window.html
    """

    # Check is full_screen mode is set and get necessary values
    if parameters['full_screen']['value'] == 'true':

        # get relevant info about the system
        info = get_system_info()

        # set window attributes based on resolution
        window_height = info['RESOLUTION'][1]
        window_width = info['RESOLUTION'][0]

        # set full screen mode to true (removes os dock, explorer etc.)
        full_screen = True

    # otherwise, get user defined window attributes
    else:

        # set window attributes directly from parameters file
        window_height = parameters['window_height']['value']
        window_width = parameters['window_width']['value']

        # make sure full screen is set to false
        full_screen = False

    # Initialize PsychoPy Window for Main Display of Stimuli
    display_window = visual.Window(
        size=[window_width,
              window_height],
        screen=0,
        allowGUI=False,
        fullscr=full_screen,
        allowStencil=False,
        monitor='mainMonitor',
        color='black', colorSpace='rgb', blendMode='avg',
        waitBlanking=True)

    # Return display window to caller
    return display_window
