# -*- coding: utf-8 -*-

import pyglet
import utility.gui_fx as gui_fx
from utils.convert import convert_to_height, convert_to_width

', main_window_width',' Initialize main window and add via gui_fx. '''
main_window = gui_fx.MenuWindow(0, 'Brain Computer Interface')
gui_fx.add_window(main_window)

# Get needed variables from window
main_window_width = main_window.width
main_window_height = main_window.height
main_window_width_half = int(main_window_width / 2)
main_window_height_half = int(main_window_height / 2)


''' Register Buttons. '''

# RSVP Keyboard button
gui_fx.add_button(
    main_window_width_half,
    main_window_height_half - convert_to_height(50, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (62, 161, 232, 255), (255, 236, 160, 255),
    'RSVP Keyboard',
    0, functionCall="set_exp_type", functionArg=['RSVPKeyboard', main_window],
    textSize=convert_to_width(12, main_window_width),
)

# Shuffle Speller button
gui_fx.add_button(
    main_window_width_half - convert_to_width(200, main_window_width),
    main_window_height_half - convert_to_height(50, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (239, 146, 40, 255), (255, 190, 117, 255),
    'Shuffle Speller', 0, functionCall="set_exp_type",
    functionArg=['Shuffle', main_window],
    textSize=convert_to_width(12, main_window_width),
)

# Matrix Speller Button
gui_fx.add_button(
    main_window_width_half + convert_to_width(200, main_window_width),
    main_window_height_half - convert_to_height(50, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (117, 173, 48, 255), (186, 232, 129, 255), 'Matrix',
    0, functionCall="set_exp_type", functionArg=['Matrix', main_window],
    textSize=convert_to_width(12, main_window_width),
)


''' Register text. '''

# Title
gui_fx.add_text(
    main_window_width_half, main_window_height_half + convert_to_height(200, main_window_height),
    (247, 247, 247, 255), convert_to_width(25, main_window_width),
    "Brain-Computer Interface", 0
)

# Help text
gui_fx.add_text(
    main_window_width_half, main_window_height_half + convert_to_height(40, main_window_height),
    (247, 247, 247, 255), convert_to_width(18, main_window_width), "Select Experiment Type:",
    0
)


''' Register images. '''

# OHSU
gui_fx.add_image(
    main_window_width_half + convert_to_width(260, main_window_width),
    main_window_height_half + convert_to_height(140, main_window_height),
    "static/images/OHSU-RGB-4C-REV.png", 0,
    float(convert_to_width(39, main_window_width),), float(convert_to_height(67, main_window_height)), False
)

# NEU
gui_fx.add_image(
    main_window_width_half - convert_to_width(305, main_window_width),
    main_window_height_half + convert_to_height(115, main_window_height),
    "static/images/northeasternuniversity_logoseal.png", 0,
    float(convert_to_width(87, main_window_width),), float(convert_to_height(88, main_window_height)), False
)

if __name__ == '__main__':
    pyglet.app.run()
