# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import visual
from utils.get_system_info import get_system_info


def init_display_window(parameters):
    """
    Init Display Window.

    Function to Initialize main display window
        needed for all later stimuli presentation.

    See Psychopy official documentation for more information:
        http://www.psychopy.org/api/visual/window.html
    """

    if parameters['full_screen']['value'] == 'true':
        info = get_system_info()
        window_height = info['RESOLUTION'][1]
        window_width = info['RESOLUTION'][0]
        full_screen = True
    else:
        window_height = parameters['window_height']['value']
        window_width = parameters['window_width']['value']
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

    # Return it to caller
    return display_window
