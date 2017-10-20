import pyglet

import utility.gui_fx as gui_fx


main_window = gui_fx.MenuWindow(0, 'Brain Computer Interface')
gui_fx.add_window(main_window)

main_window_width = main_window.width
main_window_height = main_window.height


main_window_width_half = int(main_window_width / 2)
main_window_height_half = int(main_window_height / 2)


def covert_to_height(input_number):
    return int(((input_number) / 480.0) * main_window_height)


def covert_to_width(input_number):
    return int(((input_number) / 640.0) * main_window_width)

counterbci = 0
counteradv = 0


# Register all the buttons for all the windows here.
# registering a button with a given position, size, color scheme, caption, etc.
# with a given display window causes it to be shown and clickable when that
# window is opened.
# buttons can open another window if the window is declared as the window to
# open, and/or call a function with given arguments.
# windows do not need to be declared anywhere before being opened.
# parameters: x center pos (int), y center pos (int), width (int), height (int)
# text color (tuple), button color (tuple), outline color (tuple),
# caption (str or unicode), display window (int), window to open (int),
# function name to call (str), function arguments (list), text size (int),
# scroll bar id (int)


# RSVP Keyboard button
gui_fx.add_button(
    main_window_width_half,
    main_window_height_half - covert_to_height(50),
    covert_to_width(150), covert_to_height(90),
    (25, 20, 1, 255), (62, 161, 232, 255), (255, 236, 160, 255),
    'RSVP Keyboard',
    0, functionCall="set_exp_type", functionArg=['RSVPKeyboard', main_window],
    textSize=covert_to_width(12)
)

# Shuffle Speller button
gui_fx.add_button(
    main_window_width_half - covert_to_width(200),
    main_window_height_half - covert_to_height(50),
    covert_to_width(150), covert_to_height(90),
    (25, 20, 1, 255), (239, 146, 40, 255), (255, 190, 117, 255),
    'Shuffle Speller', 0, functionCall="set_exp_type",
    functionArg=['Shuffle', main_window],
    textSize=covert_to_width(12)
)

# Matrix Speller Button
gui_fx.add_button(
    main_window_width_half + covert_to_width(200),
    main_window_height_half - covert_to_height(50),
    covert_to_width(150), covert_to_height(90),
    (25, 20, 1, 255), (117, 173, 48, 255), (186, 232, 129, 255), 'Matrix',
    0, functionCall="set_exp_type", functionArg=['Matrix', main_window],
    textSize=covert_to_width(12)
)


# register text to be displayed here.
# This is text displayed on the screen. Position and text size should be relative
# to window size.
# paramenters, x position, y position, color, size, text, display window, scroll
# bar id (if any)

gui_fx.add_text(
    main_window_width_half, main_window_height_half + covert_to_height(40),
    (247, 247, 247, 255), covert_to_width(18), "Select Experiment Type:",
    0
)
gui_fx.add_text(
    main_window_width_half, main_window_height_half + covert_to_height(200),
    (247, 247, 247, 255), covert_to_width(25),
    "Brain-Computer Interface", 0
)


# register images.
# Position, width, and height should be relative to the window size.
# parameters: x position, y position, file name, display window, width, height
gui_fx.add_image(
    main_window_width_half + covert_to_width(260),
    main_window_height_half + covert_to_height(140),
    "static/images/OHSU-RGB-4C-REV.png", 0,
    float(covert_to_width(39)), float(covert_to_height(67)), False
)
gui_fx.add_image(
    main_window_width_half - covert_to_width(305),
    main_window_height_half + covert_to_height(115),
    "static/images/northeasternuniversity_logoseal.png", 0,
    float(covert_to_width(87)), float(covert_to_height(88)), False
)

if __name__ == '__main__':
    pyglet.app.run()
