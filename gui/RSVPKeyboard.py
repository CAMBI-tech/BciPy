
from codecs import open as codecsopen
from json import load as jsonload

from sys import path
from warnings import warn

import pyglet

path.append('utils/')
import gui_fx

path.append('io/')
from load import load_json_parameters

main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
gui_fx.add_window(main_window)

main_window_width = main_window.width
main_window_height = main_window.height

# declare scroll bars.
# parameters: bar class(height of window), id
window_three_bar = gui_fx.ScrollBar(main_window_height, 3)
window_zero_bar = gui_fx.ScrollBar(main_window_height, 0, visible=False)
window_four_bar = gui_fx.ScrollBar(main_window_height, 4)
option_tab_bar = gui_fx.ScrollBar(main_window_width, 100,
                                  visible=False,
                                  horizontal=True)

main_window_width_half = int(main_window_width / 2)
main_window_height_half = int(main_window_height / 2)


def covert_to_height(input_number):
    return int(((input_number) / 480.0) * main_window_height)


def covert_to_width(input_number):
    return int(((input_number) / 640.0) * main_window_width)

# Scrolling content
# Giving an item a scroll bar id causes the program to calculate its position
# based on the position of the scroll bar. The y position should represent the
# item's actual height before scrolling, and should still be relative to the
# height of the window.
# The scroll bar's content height is the height of the items scrolled by the bar.
# This is used to calculate how much they should scroll relative to the
# movement of the bar.

gui_fx.add_text(
    main_window_width_half + covert_to_width(10),
    main_window_height - covert_to_height(20),
    (247, 247, 247, 255), covert_to_width(20), "Parameters", 3, 3
)
gui_fx.add_text(
    main_window_width_half + covert_to_width(10),
    main_window_height - covert_to_height(20),
    (247, 247, 247, 255), covert_to_width(20), "Advanced Options", 4, 4
)
window_three_bar.addToContentHeight(60)
window_four_bar.addToContentHeight(60)

path = "parameters/parameters.json"
file_data = load_json_parameters(path)

counterbci = 0
counteradv = 0

# values_array contains the names of all the values in the config file, so that
# those names can be passed to the save/load data functions called by buttons.
values_array = []
for json_item in file_data:
    section = file_data[json_item]["section"]
    section_boolean = section == 'bci_config'
    display_window = 3 if file_data[json_item]["section"] == 'bci_config' \
        else 4
    section_counter = counterbci if (section_boolean) else counteradv
    section_string = 'bci_config' if (section_boolean) else 'advanced_config'
    if (section_boolean):
        counterbci = counterbci + 1
    else:
        counteradv = counteradv + 1
    readable_caption = file_data[json_item]["readableName"]
    isNumeric = file_data[json_item]["isNumeric"]
    # adds name of each parameter above its input box
    gui_fx.add_text(
        main_window_width_half + covert_to_width(10),
        covert_to_height((section_counter) - (window_three_bar.contentHeight
                         if (section_boolean)
                         else window_four_bar.contentHeight)) +
        main_window_height,
        (247, 247, 247, 255), covert_to_width(9), readable_caption,
        display_window, display_window
    )
    if (section_boolean):
        window_three_bar.addToContentHeight(35)
    else:
        window_four_bar.addToContentHeight(35)

    # adds help button for each parameter
    gui_fx.add_button(
        main_window_width_half + covert_to_width(220),
        covert_to_height((section_counter) - (window_three_bar.contentHeight
                         if (section_boolean)
                         else window_four_bar.contentHeight)) + main_window_height,
        covert_to_width(20), covert_to_height(20),
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '?',
        display_window, functionCall="display_help_pointers",
        functionArg=(path, json_item),
        scrollBar=display_window, textSize=covert_to_width(12)
    )
    value = file_data[json_item]["value"]
    if(value == 'true' or value == 'false'):
        valueBoolean = True if (value == 'true') else False

        # adds a switch instead of the input box for the parameter if it is a boolean
        gui_fx.add_switch(
            (gui_fx.BooleanSwitch(
                main_window_width_half + covert_to_width(10),
                covert_to_height((section_counter) - (window_three_bar.contentHeight
                                 if (section_boolean)
                                 else window_four_bar.contentHeight)) + main_window_height,
                covert_to_width(200), covert_to_height(38),
                json_item, valueBoolean, covert_to_width(19),
                section_string
            ), display_window, display_window)
        )
    else:
        # Adds an input field if an input field is needed
        gui_fx.add_input(
            gui_fx.InputField(json_item, section_string, True
                              if isNumeric == "true"
                              else False),
            main_window_width_half + covert_to_width(10),
            covert_to_height((section_counter) - (window_three_bar.contentHeight
                             if (section_boolean)
                             else window_four_bar.contentHeight)) + main_window_height,
            covert_to_width(300), covert_to_height(40),
            display_window, covert_to_width(10), display_window
        )
        gui_fx.inputFields[len(gui_fx.inputFields) - 1][0].text = value

        # adds a drop-down list of recommended values for a parameter if needed
        if(file_data[json_item]["recommended_values"] != ''):
            gui_fx.add_button(
                main_window_width_half + covert_to_width(185),
                covert_to_height((section_counter) - (window_three_bar.contentHeight
                                 if (section_boolean)
                                 else window_four_bar.contentHeight)) + main_window_height,
                covert_to_width(30), covert_to_height(30),
                (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '',
                display_window, functionCall="drop_items",
                functionArg=[json_item, display_window, path, "recommended_values"],
                scrollBar=display_window, textSize=covert_to_width(22)
            )
            gui_fx.add_image(
                main_window_width_half + covert_to_width(172),
                covert_to_height((section_counter) - (window_three_bar.contentHeight
                                 if (section_boolean)
                                 else window_four_bar.contentHeight) - 15) + main_window_height,
                "static/images/triangle.png", display_window,
                float(covert_to_width(25)), float(covert_to_height(25)),
                display_window
            )
    values_array.append(json_item)
    if (section_boolean):
        window_three_bar.addToContentHeight(35)
    else:
        window_four_bar.addToContentHeight(35)


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
# Extend options menu
gui_fx.add_button(
    main_window_width_half - covert_to_width(235),
    main_window_height_half + covert_to_height(210),
    covert_to_width(100), covert_to_height(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Show Options', 3,
    functionCall="move_menu", functionArg=[100, covert_to_width(90)],
    textSize=covert_to_width(8)
)

# Presentation mode button
gui_fx.add_button(
    main_window_width_half, main_window_height_half + covert_to_height(50),
    covert_to_width(400), covert_to_height(75), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Presentation Mode', 2,
    functionCall="run_python_file", functionArg=['testing/testfile.py'],
    textSize=covert_to_width(20)
)
# View signals button- path the executable for viewing quality of signals
# gui_fx.add_button(
#     main_window_width_half, main_window_height_half - covert_to_height(50),
#     covert_to_width(400), covert_to_height(75), (40, 40, 40, 255),
#     (219, 219, 219, 255), (89, 89, 89, 255), 'View Signals', 2,
#     functionCall="run_executable",
#     functionArg=[environ['USERPROFILE'] + "\\Desktop", 'exe_name', True],
#     textSize=covert_to_width(20)
# )

# Configure parameters button
gui_fx.add_button(
    main_window_width_half - covert_to_width(155),
    main_window_height_half - covert_to_height(150),
    covert_to_width(300), covert_to_height(70),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255),
    'Configure Parameters', 0, 3, textSize=covert_to_width(16)
)

# Save values button
gui_fx.add_button(
    0,
    main_window_height_half - covert_to_height(100),
    covert_to_width(150), covert_to_height(60), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Save Values', 3,
    functionCall="writeValuesToFile",
    functionArg=(['bci_config', 'advanced_config'], values_array),
    textSize=covert_to_width(16), scrollBar=100
)

# Load values button
gui_fx.add_button(
    0,
    main_window_height_half - covert_to_height(170),
    covert_to_width(150), covert_to_height(60), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Load Values', 3,
    functionCall="read_values_from_file",
    functionArg=(['bci_config', 'advanced_config'], values_array),
    textSize=covert_to_width(16), scrollBar=100
)

# Advanced options button
gui_fx.add_button(
    0,
    main_window_height_half + covert_to_height(30),
    covert_to_width(150), covert_to_height(50), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Advanced Options', 3, 4,
    textSize=covert_to_width(10), scrollBar=100
)

# Free spell button
gui_fx.add_button(
    main_window_width_half, main_window_height_half - covert_to_height(40),
    covert_to_width(100), covert_to_height(90),
    (25, 20, 1, 255), (239, 212, 105, 255), (255, 236, 160, 255), 'Free Spell',
    0, functionCall="set_trial_type", functionArg=[3],
    textSize=covert_to_width(12)
)

# FRP Calibration button
gui_fx.add_button(
    main_window_width_half - covert_to_width(110),
    main_window_height_half - covert_to_height(40),
    covert_to_width(100), covert_to_height(90),
    (25, 20, 1, 255), (239, 146, 40, 255), (255, 190, 117, 255),
    'FRP Calibration', 0, functionCall="set_trial_type", functionArg=[2],
    textSize=covert_to_width(12)
)

# Copy phrase button
gui_fx.add_button(
    main_window_width_half + covert_to_width(110),
    main_window_height_half - covert_to_height(40),
    covert_to_width(100), covert_to_height(90),
    (25, 20, 1, 255), (117, 173, 48, 255), (186, 232, 129, 255), 'Copy Phrase',
    0, functionCall="set_trial_type", functionArg=[4],
    textSize=covert_to_width(12)
)

# ERP calibration button
gui_fx.add_button(
    main_window_width_half - covert_to_width(220),
    main_window_height_half - covert_to_height(40),
    covert_to_width(100), covert_to_height(90),
    (25, 20, 1, 255), (221, 37, 56, 255), (245, 101, 71, 255),
    'ERP Calibration', 0, functionCall="set_trial_type", functionArg=[1],
    textSize=covert_to_width(12)
)

# Mastery task button
gui_fx.add_button(
    main_window_width_half + covert_to_width(220),
    main_window_height_half - covert_to_height(40),
    covert_to_width(100), covert_to_height(90),
    (25, 20, 1, 255), (62, 161, 232, 255), (81, 217, 255, 255), 'Mastery Task',
    0, functionCall="set_trial_type", functionArg=[5],
    textSize=covert_to_width(12)
)

# Drop-down list button for user ids
gui_fx.add_button(
    main_window_width_half + covert_to_width(122),
    main_window_height_half + covert_to_height(100),
    covert_to_width(40), covert_to_height(40),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0,
    functionCall="drop_items", functionArg=['user_id', 0, "users.txt", False],
    textSize=covert_to_width(22)
)

# Calculate AUC button
gui_fx.add_button(
    main_window_width_half + covert_to_width(155),
    main_window_height_half - covert_to_height(150),
    covert_to_width(300), covert_to_height(70),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), 'Calculate AUC',
    0, functionCall="run_python_file", functionArg=['testing/testfile.py'],
    textSize=covert_to_width(16)
)

# Search parameters button
gui_fx.add_button(
    0,
    main_window_height_half + covert_to_height(90),
    covert_to_width(60), covert_to_height(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Search', 3,
    functionCall="search_parameters", functionArg=[path, 3, 'search'],
    textSize=covert_to_width(8), scrollBar=100
)

# Search advanced parameters button
gui_fx.add_button(
    main_window_width_half - covert_to_width(230),
    main_window_height_half + covert_to_height(90),
    covert_to_width(60), covert_to_height(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Search', 4,
    functionCall="search_parameters", functionArg=[path, 4, 'advancedsearch'],
    textSize=covert_to_width(8)
)

# Retract options menu
gui_fx.add_button(
    main_window_width_half - covert_to_width(325),
    main_window_height_half + covert_to_height(210),
    covert_to_width(100), covert_to_height(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Hide Options', 3,
    functionCall="move_menu", functionArg=[100, covert_to_width(90)],
    textSize=covert_to_width(8), scrollBar=100
)
option_tab_bar.addToContentHeight(20)

# register all the input text fields for all the windows here.
# InputFields are passed a name as a parameter, which is used as a field name
# when the contents of the field are written to a text file.
# Like buttons, the width, height, and positions of input fields should be
# relative to the screen size. Input fields display in the window with their
# given display window id.
# parameters: inputField data, x center pos, y center pos, width, height, display
# window, text size, scroll bar id (if any)
# User id input field
gui_fx.add_input(
    gui_fx.InputField('user_id', False, False), main_window_width_half,
    main_window_height_half + covert_to_height(100),
    covert_to_width(300), covert_to_height(50), 0,
    covert_to_width(14)
)
# main parameters search menu
gui_fx.add_input(
    gui_fx.InputField('search', False, False), 0,
    main_window_height_half + covert_to_height(130), covert_to_width(150), covert_to_height(40), 3,
    covert_to_width(10), scrollBar=100
)

# advanced parameters search menu
gui_fx.add_input(
    gui_fx.InputField('advancedsearch', False, False), main_window_width_half - covert_to_width(230),
    main_window_height_half + covert_to_height(130), covert_to_width(150), covert_to_height(40), 4,
    covert_to_width(10)
)

# register text to be displayed here.
# This is text displayed on the screen. Position and text size should be relative
# to window size.
# paramenters, x position, y position, color, size, text, display window, scroll
# bar id (if any)
gui_fx.add_text(
    main_window_width_half, main_window_height_half + covert_to_height(150),
    (247, 247, 247, 255), covert_to_width(18),
    "Enter or select a user ID:", 0
)
gui_fx.add_text(
    main_window_width_half, main_window_height_half + covert_to_height(200),
    (247, 247, 247, 255), covert_to_width(18),
    "RSVP Keyboard", 0
)
gui_fx.add_text(
    main_window_width_half, main_window_height_half + covert_to_height(40),
    (247, 247, 247, 255), covert_to_width(18), "Select type of trial:", 0
)
gui_fx.add_text(
    main_window_width_half, main_window_height_half + covert_to_height(150),
    (247, 247, 247, 255), covert_to_width(21), "Select Mode:", 2
)
gui_fx.add_text(
    0, main_window_height_half + covert_to_height(170),
    (247, 247, 247, 255), covert_to_width(11), "Search Parameters", 3, scrollBar=100
)
gui_fx.add_text(
    main_window_width_half - covert_to_width(230), main_window_height_half + covert_to_height(160),
    (247, 247, 247, 255), covert_to_width(8), "Search Advanced Parameters", 4
)

# register images.
# Position, width, and height should be relative to the window size.
# parameters: x position, y position, file name, display window, width, height, scroll bar
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
gui_fx.add_image(
    main_window_width_half + covert_to_width(105),
    main_window_height_half + (80/480.0)*main_window_height,
    "static/images/triangle.png", 0,
    float(covert_to_width(33)), float(covert_to_height(33)), False
)

#real scroll bar registration
#add_scroll takes the scroll bar itself and the id of the attached window as
#parameters.
gui_fx.add_scroll((window_three_bar, 3))
gui_fx.add_scroll((window_zero_bar, 0))
gui_fx.add_scroll((window_four_bar, 4))
gui_fx.add_scroll((option_tab_bar, 3))

if __name__ == '__main__':
    pyglet.app.run()
