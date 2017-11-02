import pyglet
import utility.gui_fx as gui_fx
from helpers.load import load_json_parameters
from utils.convert import convert_to_height, convert_to_width

# Initialize main window and add via gui_fx
main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
gui_fx.add_window(main_window)

# Get needed variables from window
main_window_width = main_window.width
main_window_height = main_window.height
main_window_width_half = int(main_window_width / 2)
main_window_height_half = int(main_window_height / 2)


''' Register Scroll Bars '''
window_three_bar = gui_fx.ScrollBar(main_window_height, 3)
window_zero_bar = gui_fx.ScrollBar(main_window_height, 0, visible=False)
window_four_bar = gui_fx.ScrollBar(main_window_height, 4)
option_tab_bar = gui_fx.ScrollBar(main_window_width, 100,
                                  visible=False,
                                  horizontal=True)

''' Register Scrolling content'''
gui_fx.add_text(
    main_window_width_half + convert_to_width(10, main_window_width),
    main_window_height - convert_to_height(20, main_window_height),
    (247, 247, 247, 255), convert_to_width(20, main_window_width), "Parameters", 3, 3
)
gui_fx.add_text(
    main_window_width_half + convert_to_width(10, main_window_width),
    main_window_height - convert_to_height(20, main_window_height),
    (247, 247, 247, 255), convert_to_width(20, main_window_width), "Advanced Options", 4, 4
)
window_three_bar.addToContentHeight(60)
window_four_bar.addToContentHeight(60)

''' Register Parameters '''
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
        main_window_width_half + convert_to_width(10, main_window_width),
        convert_to_height((section_counter) - (window_three_bar.contentHeight
                         if (section_boolean)
                         else window_four_bar.contentHeight), main_window_height) +
        main_window_height,
        (247, 247, 247, 255), convert_to_width(9, main_window_width), readable_caption,
        display_window, display_window
    )
    if (section_boolean):
        window_three_bar.addToContentHeight(35)
    else:
        window_four_bar.addToContentHeight(35)

    # adds help button for each parameter
    gui_fx.add_button(
        main_window_width_half + convert_to_width(220, main_window_width),
        convert_to_height((section_counter) - (window_three_bar.contentHeight
                         if (section_boolean)
                         else window_four_bar.contentHeight), main_window_height) + main_window_height,
        convert_to_width(20, main_window_width), convert_to_height(20, main_window_height),
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '?',
        display_window, functionCall="display_help_pointers",
        functionArg=(path, json_item),
        scrollBar=display_window, textSize=convert_to_width(12, main_window_width)
    )
    value = file_data[json_item]["value"]
    if(value == 'true' or value == 'false'):
        valueBoolean = True if (value == 'true') else False

        # adds a switch instead of the input box for the parameter if it is a boolean
        gui_fx.add_switch(
            (gui_fx.BooleanSwitch(
                main_window_width_half + convert_to_width(10, main_window_width),
                convert_to_height((section_counter) - (window_three_bar.contentHeight
                                 if (section_boolean)
                                 else window_four_bar.contentHeight), main_window_height) + main_window_height,
                convert_to_width(200, main_window_width), convert_to_height(38, main_window_height),
                json_item, valueBoolean, convert_to_width(19, main_window_width),
                section_string
            ), display_window, display_window)
        )
    else:
        # Adds an input field if an input field is needed
        gui_fx.add_input(
            gui_fx.InputField(json_item, section_string, True
                              if isNumeric == "true"
                              else False),
            main_window_width_half + convert_to_width(10, main_window_width),
            convert_to_height((section_counter) - (window_three_bar.contentHeight
                             if (section_boolean)
                             else window_four_bar.contentHeight), main_window_height) + main_window_height,
            convert_to_width(300, main_window_width), convert_to_height(40, main_window_height),
            display_window, convert_to_width(10, main_window_width), display_window
        )
        gui_fx.inputFields[len(gui_fx.inputFields) - 1][0].text = value

        # adds a drop-down list of recommended values for a parameter if needed
        if(file_data[json_item]["recommended_values"] != ''):
            gui_fx.add_button(
                main_window_width_half + convert_to_width(185, main_window_width),
                convert_to_height((section_counter) - (window_three_bar.contentHeight
                                 if (section_boolean)
                                 else window_four_bar.contentHeight), main_window_height) + main_window_height,
                convert_to_width(30, main_window_width), convert_to_height(30, main_window_height),
                (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '',
                display_window, functionCall="drop_items",
                functionArg=[json_item, display_window, path, "recommended_values"],
                scrollBar=display_window, textSize=convert_to_width(22, main_window_width)
            )
            gui_fx.add_image(
                main_window_width_half + convert_to_width(172, main_window_width),
                convert_to_height((section_counter) - (window_three_bar.contentHeight
                                 if (section_boolean)
                                 else window_four_bar.contentHeight) - 15, main_window_height) + main_window_height,
                "static/images/triangle.png", display_window,
                float(convert_to_width(25, main_window_width)), float(convert_to_height(25, main_window_height)),
                display_window
            )
    values_array.append(json_item)
    if (section_boolean):
        window_three_bar.addToContentHeight(35)
    else:
        window_four_bar.addToContentHeight(35)


''' Register all the buttons for all the windows here.'''

# Extend options menu
gui_fx.add_button(
    main_window_width_half - convert_to_width(235, main_window_width),
    main_window_height_half + convert_to_height(210, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(30, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Show Options', 3,
    functionCall="move_menu", functionArg=[100, convert_to_width(90, main_window_width)],
    textSize=convert_to_width(8, main_window_width)
)

# Presentation mode button
gui_fx.add_button(
    main_window_width_half, main_window_height_half + convert_to_height(50, main_window_height),
    convert_to_width(400, main_window_width), convert_to_height(75, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Presentation Mode', 2,
    functionCall="exec_bci_main", functionArg=[file_data, main_window, "RSVP"],
    textSize=convert_to_width(20, main_window_width)
)
# View signals button- path the executable for viewing quality of signals
# gui_fx.add_button(
#     main_window_width_half, main_window_height_half - convert_to_height(50, main_window_height),
#     convert_to_width(400, main_window_width), convert_to_height(75, main_window_height), (40, 40, 40, 255),
#     (219, 219, 219, 255), (89, 89, 89, 255), 'View Signals', 2,
#     functionCall="run_executable",
#     functionArg=[environ['USERPROFILE'] + "\\Desktop", 'exe_name', True],
#     textSize=convert_to_width(20, main_window_width)
# )

# Configure parameters button
gui_fx.add_button(
    main_window_width_half - convert_to_width(155, main_window_width),
    main_window_height_half - convert_to_height(150, main_window_height),
    convert_to_width(300, main_window_width), convert_to_height(70, main_window_height),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255),
    'Configure Parameters', 0, 3, textSize=convert_to_width(16, main_window_width)
)

# Save values button
gui_fx.add_button(
    0,
    main_window_height_half - convert_to_height(100, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(60, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Save Values', 3,
    functionCall="writeValuesToFile",
    functionArg=(['bci_config', 'advanced_config'], values_array),
    textSize=convert_to_width(16, main_window_width), scrollBar=100
)

# Load values button
gui_fx.add_button(
    0,
    main_window_height_half - convert_to_height(170, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(60, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Load Values', 3,
    functionCall="read_values_from_file",
    functionArg=(['bci_config', 'advanced_config'], values_array),
    textSize=convert_to_width(16, main_window_width), scrollBar=100
)

# Advanced options button
gui_fx.add_button(
    0,
    main_window_height_half + convert_to_height(30, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(50, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Advanced Options', 3, 4,
    textSize=convert_to_width(10, main_window_width), scrollBar=100
)

# Free spell button
gui_fx.add_button(
    main_window_width_half, main_window_height_half - convert_to_height(40, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (239, 212, 105, 255), (255, 236, 160, 255), 'Free Spell',
    0, functionCall="set_trial_type", functionArg=[3],
    textSize=convert_to_width(12, main_window_width)
)

# FRP Calibration button
gui_fx.add_button(
    main_window_width_half - convert_to_width(110, main_window_width),
    main_window_height_half - convert_to_height(40, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (239, 146, 40, 255), (255, 190, 117, 255),
    'FRP Calibration', 0, functionCall="set_trial_type", functionArg=[2],
    textSize=convert_to_width(12, main_window_width)
)

# Copy phrase button
gui_fx.add_button(
    main_window_width_half + convert_to_width(110, main_window_width),
    main_window_height_half - convert_to_height(40, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (117, 173, 48, 255), (186, 232, 129, 255), 'Copy Phrase',
    0, functionCall="set_trial_type", functionArg=[4],
    textSize=convert_to_width(12, main_window_width)
)

# ERP calibration button
gui_fx.add_button(
    main_window_width_half - convert_to_width(220, main_window_width),
    main_window_height_half - convert_to_height(40, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (221, 37, 56, 255), (245, 101, 71, 255),
    'ERP Calibration', 0, functionCall="set_trial_type", functionArg=[1],
    textSize=convert_to_width(12, main_window_width)
)

# Mastery task button
gui_fx.add_button(
    main_window_width_half + convert_to_width(220, main_window_width),
    main_window_height_half - convert_to_height(40, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(90, main_window_height),
    (25, 20, 1, 255), (62, 161, 232, 255), (81, 217, 255, 255), 'Mastery Task',
    0, functionCall="set_trial_type", functionArg=[5],
    textSize=convert_to_width(12, main_window_width)
)

# Drop-down list button for user ids
gui_fx.add_button(
    main_window_width_half + convert_to_width(122, main_window_width),
    main_window_height_half + convert_to_height(100, main_window_height),
    convert_to_width(40, main_window_width), convert_to_height(40, main_window_height),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0,
    functionCall="drop_items", functionArg=['user_id', 0, "users.txt", False],
    textSize=convert_to_width(22, main_window_width)
)

# Calculate AUC button
gui_fx.add_button(
    main_window_width_half + convert_to_width(155, main_window_width),
    main_window_height_half - convert_to_height(150, main_window_height),
    convert_to_width(300, main_window_width), convert_to_height(70, main_window_height),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), 'Calculate AUC',
    0, functionCall="run_python_file", functionArg=['gui/tests/testfile.py'],
    textSize=convert_to_width(16, main_window_width)
)

# Back to BCI Main Button
gui_fx.add_button(
    main_window_width_half - convert_to_width(250, main_window_width),
    main_window_height_half - convert_to_height(220, main_window_height),
    convert_to_width(150, main_window_width),
    convert_to_height(25, main_window_height),
   (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), ' <<< BCI Main',
    0, functionCall="bci_main_exec", functionArg=[main_window],
    textSize=convert_to_width(7, main_window_width),
)

# Search parameters button
gui_fx.add_button(
    0,
    main_window_height_half + convert_to_height(90, main_window_height),
    convert_to_width(60, main_window_width), convert_to_height(30, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Search', 3,
    functionCall="search_parameters", functionArg=[path, 3, 'search'],
    textSize=convert_to_width(8, main_window_width), scrollBar=100
)

# Search advanced parameters button
gui_fx.add_button(
    main_window_width_half - convert_to_width(230, main_window_width),
    main_window_height_half + convert_to_height(90, main_window_height),
    convert_to_width(60, main_window_width), convert_to_height(30, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Search', 4,
    functionCall="search_parameters", functionArg=[path, 4, 'advancedsearch'],
    textSize=convert_to_width(8, main_window_width)
)

# Retract options menu
gui_fx.add_button(
    main_window_width_half - convert_to_width(325, main_window_width),
    main_window_height_half + convert_to_height(210, main_window_height),
    convert_to_width(100, main_window_width), convert_to_height(30, main_window_height), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Hide Options', 3,
    functionCall="move_menu", functionArg=[100, convert_to_width(90, main_window_width)],
    textSize=convert_to_width(8, main_window_width), scrollBar=100
)
option_tab_bar.addToContentHeight(20)

''' Register input text fields'''
# User Id
gui_fx.add_input(
    gui_fx.InputField('user_id', False, False), main_window_width_half,
    main_window_height_half + convert_to_height(100, main_window_height),
    convert_to_width(300, main_window_width), convert_to_height(50, main_window_height), 0,
    convert_to_width(14, main_window_width)
)
# Main parameters search menu
gui_fx.add_input(
    gui_fx.InputField('search', False, False), 0,
    main_window_height_half + convert_to_height(130, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(40, main_window_height), 3,
    convert_to_width(10, main_window_width), scrollBar=100
)

# Advanced parameters search menu
gui_fx.add_input(
    gui_fx.InputField('advancedsearch', False, False),
    main_window_width_half - convert_to_width(230, main_window_width),
    main_window_height_half + convert_to_height(130, main_window_height),
    convert_to_width(150, main_window_width), convert_to_height(40, main_window_height), 4,
    convert_to_width(10,main_window_width)
)

''' Register text. '''

# Select ID help text
gui_fx.add_text(
    main_window_width_half, main_window_height_half + convert_to_height(150, main_window_height),
    (247, 247, 247, 255), convert_to_width(18, main_window_width),
    "Enter or select a user ID:", 0
)

# Title
gui_fx.add_text(
    main_window_width_half, main_window_height_half + convert_to_height(200, main_window_height),
    (247, 247, 247, 255), convert_to_width(18, main_window_width),
    "RSVP Keyboard", 0
)

# Help text for trial
gui_fx.add_text(
    main_window_width_half, main_window_height_half + convert_to_height(40, main_window_height),
    (247, 247, 247, 255), convert_to_width(18, main_window_width), "Select type of trial:", 0
)

# Help text for mode selection
gui_fx.add_text(
    main_window_width_half, main_window_height_half + convert_to_height(150, main_window_height),
    (247, 247, 247, 255), convert_to_width(21, main_window_width), "Select Mode:", 2
)

# Search Parameters help text
gui_fx.add_text(
    0, main_window_height_half + convert_to_height(170, main_window_height),
    (247, 247, 247, 255), convert_to_width(11,main_window_width), "Search Parameters", 3, scrollBar=100
)

# Search Advanced Parameters help text
gui_fx.add_text(
    main_window_width_half - convert_to_width(230, main_window_width),
    main_window_height_half + convert_to_height(160, main_window_height),
    (247, 247, 247, 255), convert_to_width(8, main_window_width), "Search Advanced Parameters", 4
)

''' Register images. '''

# OHSU
gui_fx.add_image(
    main_window_width_half + convert_to_width(260, main_window_width),
    main_window_height_half + convert_to_height(140, main_window_height),
    "static/images/OHSU-RGB-4C-REV.png", 0,
    float(convert_to_width(39, main_window_width)), float(convert_to_height(67, main_window_height)), False
)

# NEU
gui_fx.add_image(
    main_window_width_half - convert_to_width(305, main_window_width),
    main_window_height_half + convert_to_height(115, main_window_height),
    "static/images/northeasternuniversity_logoseal.png", 0,
    float(convert_to_width(87, main_window_width)), float(convert_to_height(88, main_window_height)), False
)

# Select users button (near the user id input field)
gui_fx.add_image(
    main_window_width_half + convert_to_width(105, main_window_width),
    main_window_height_half + (80/480.0)*main_window_height,
    "static/images/triangle.png", 0,
    float(convert_to_width(33, main_window_width)), float(convert_to_height(33, main_window_height)), False
)

''' Scroll bar registration'''
gui_fx.add_scroll((window_three_bar, 3))
gui_fx.add_scroll((window_zero_bar, 0))
gui_fx.add_scroll((window_four_bar, 4))
gui_fx.add_scroll((option_tab_bar, 3))

if __name__ == '__main__':
    pyglet.app.run()
