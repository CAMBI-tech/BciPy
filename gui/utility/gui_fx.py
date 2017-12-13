import subprocess
from codecs import open as codecsopen
from collections import OrderedDict
from inspect import getargspec
from json import dumps as jsondumps
from json import load as jsonload
from os import chdir
from os import path as ospath
from sys import executable
from warnings import warn
import bci_main
from eeg_model.offline_analysis import offline_analysis

import pyglet
import wx

import gui_fx


# arrays of buttons, windows, text, input fields, scroll bars, on the screen
buttons = []
windows = []
inputFields = []
textBoxes = []
scroll_bars = []
images = []
switches = []
horizontalScrolls = []
boxes = []
# cache of label objects
labelCache = {}

# index of buttons currently displayed on the screen
buttonsOnScreen = []

main_window_width = 640
main_window_height = 480

# which button is currently moused over, if any
mouseOnButton = 1000
# same for input fields
mouseOnInput = 1000
# which input field is currently active
currentActiveInputField = 1000
# which switch is moused over, if any
mouseOnSwitch = 1000
# shift key
shiftPressed = False
# which scroll bar is currently moused over, if any
scrollMouse = 1000
# position of the typing cursor relative to the length of the string
typingCursorPos = 0

# current help pointer string to be displayed by the mouse, if any
mouseHelp = False
# current mouse y pos
mouseY = 0
# current mouse x pos
mouseX = 0

# bci trial type
trialType = 0
# bci user id
userId = 0


# invisible wxpython window to show save file dialogs
wxApp = wx.App()
wxWindow = wx.Window()
wxWindow.Hide()

# scroll bar height
scrollBarHeight = 50


# Moves a scroll bar back and forth by a given amount.
# Used by the show/hide options buttons in the parameters window.
def move_menu(barId, moveAmount):
    for eachBar in scroll_bars:
        if(eachBar[0].scroll_id == barId):
            if(eachBar[0].relativeYPos != moveAmount):
                move_scroll_bar(barId, moveAmount)
            else:
                move_scroll_bar(barId, eachBar[0].relativeYPos - moveAmount)


# Moves a scroll bar by a given amount, adjusting both the displayed bar and
# the bar's content's position accordingly.
def move_scroll_bar(barId, moveAmount):
    for eachBar in scroll_bars:
        if(eachBar[0].scroll_id == barId):
            eachBar[0].relativeYPos = moveAmount
            bar_amount = int(
                (moveAmount * (eachBar[0].height - scrollBarHeight)) /
                ((eachBar[0].contentHeight / (640.0 if eachBar[0].isHorizontal 
                                              else 480.0)) * 
                 (main_window_width if eachBar[0].isHorizontal
                  else main_window_height)))
            eachBar[0].yPos = bar_amount


def run_offline_analysis(window, data_folder=None):
    try:
        data_folder = '/Users/scit_tab/Desktop/bci/data/test_user'
        offline_analysis(data_folder=data_folder)
    except Exception as e:
        create_message_box(
            "BCI Error",
            "Error in Offline Analysis: %s" % (e),
            wx.OK)


# sets exp type from main menu and opens the next desired gui
def set_exp_type(expType, window):
    # set textBoxes globally
    global textBoxes
    global buttons

    # RSVP
    if expType is 'RSVPKeyboard':
        # Remove all previous text and buttons
        textBoxes = []
        buttons = []

        # Run the RSVP python gui
        run_python_file('gui/RSVPKeyboard.py')

        # Close the BCI Main window
        window.close()

    # Shuffle
    elif expType is 'Shuffle':
        create_message_box("BCI Main", "Shuffle is not implemeted yet!", wx.OK)

    # Matrix
    elif expType is 'Matrix':
        create_message_box("BCI Main", "Matrix is not implemeted yet!", wx.OK)

    else:
        create_message_box("BCI Error", "Input Not Recognized", wx.OK)


# Go back to the bci main window, from any other window
def bci_main_exec(window):
    # set textBoxes, buttons, inputs, and images globally
    global textBoxes
    global buttons
    global inputFields
    global images

    # BCI Main
    try:
        # Remove all previous text, buttons, inputs and images
        textBoxes = []
        buttons = []
        inputFields = []
        images = []

        # Run the BCI main python gui
        run_python_file('gui/BCInterface.py')

        # Close the other window
        window.close()

    except:
        raise Exception('Input not recognized')

# for text boxes, inserts a given character (symbol) at the location indicated
# by typingcursorpos in the text of an input field located at inputFieldIndex 
# in the global array
def insert_symbol_at_index(symbol, inputFieldIndex):
    global typingCursorPos
    inputFields[inputFieldIndex][0].text = inputFields[inputFieldIndex][0] \
        .text[:typingCursorPos] + symbol + \
        inputFields[inputFieldIndex][0].text[typingCursorPos:]
    typingCursorPos = typingCursorPos + 1


# changes the text of an input box. for drop-downs.
def change_input_text(inputName, changedText):
    global typingCursorPos
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == inputName):
            inputFields[counter][0].text = changedText
            counter = len(inputFields)
            typingCursorPos = len(changedText) - 1


# for drop-down menus. creates a drop-down of all items in a text file.
# text_box_name is the name of the parent input field. windowId is the name of
# the window that the buttons should be located in. filename is the name of the
# file the values are read from. readValues should be true if the file is JSON
def drop_items(text_box_name, windowId, filename, readValues):
    global scroll_bars
    for counter in range(0, len(scroll_bars)):
        if(scroll_bars[counter][0].scroll_id == windowId):
            bar_index = counter
            counter = len(scroll_bars)
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == text_box_name):
            if(ospath.isfile(filename)):
                with codecsopen(filename, 'r', encoding='utf-8') as f:
                    user_array = []
                    # determines whether content should be read as a json file
                    if(readValues == False):
                        user_array = f.readlines()
                    else:
                        try:
                            file_data = jsonload(f)
                            user_array = file_data[text_box_name][readValues]
                        except ValueError:
                            warn('File ' + str(filename) + ' is an invalid JSON file.')
                    for counter2 in range(0, (len(user_array))):
                        if(isinstance(user_array[counter2], basestring)):
                            add_button(
                                inputFields[counter][1],
                                (inputFields[counter][2] - (inputFields[counter][4]) - ((counter2 - 1) * 10)) - counter2*20,
                                inputFields[counter][3],
                                int((20/480.0)*main_window_height), (40, 40, 40, 255),
                                (219, 219, 219, 255), (89, 89, 89, 255),
                                user_array[counter2].replace("\n", '').replace("'u", ''),
                                windowId, functionCall="change_input_text",
                                functionArg=[text_box_name, user_array[counter2].replace("\n", '').replace("'u", '')],
                                scrollBar=windowId,
                                textSize=int((10/640.0)*main_window_width), isTemp=True,
                                prioritizeTask=True, fontName='Arial'
                            )
                        else:
                            add_button(
                                inputFields[counter][1],
                                (inputFields[counter][2] - (inputFields[counter][4]) - ((counter2 - 1) * 10)) - counter2*20,
                                inputFields[counter][3],
                                int((20/480.0)*main_window_height), (40, 40, 40, 255),
                                (219, 219, 219, 255), (89, 89, 89, 255),
                                str(user_array[counter2]).replace("\n", '').replace("'u", ''),
                                windowId, functionCall="change_input_text",
                                functionArg=[text_box_name, str(user_array[counter2]).replace("\n", '').replace("'u", '')],
                                scrollBar=windowId,
                                textSize=int((10/640.0)*main_window_width), isTemp=True,
                                prioritizeTask=True, fontName='Arial'
                            )
                        scroll_bars[bar_index][0].addToContentHeight((10/640.0)*main_window_width)
                    f.close()
            else:
                warn("File " + str(filename) + " could not be found.")


# sets trial type global, writes user id to file if the user id is not already in the file, and opens new window
def set_trial_type(numId):
    global trialType
    global userId
    global new_window
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == 'user_id'):
            if(inputFields[counter][0].text != ''):
                trialType = numId
                userId = inputFields[counter][0].text
                new_window = MenuWindow(2, ' ')
                add_window(new_window)
                if(ospath.isfile("users.txt")):
                    with codecsopen("users.txt", 'r', encoding='utf-8') as f:
                        user_array = f.readlines()
                        for eachItem in user_array:
                            if(eachItem.replace("\n", '').replace("'u", '') == userId):
                                f.close()
                                return
                    f.close()
                    with codecsopen("users.txt", 'a', encoding='utf-8') as f:
                        f.write(str(userId) + "\n")
                    f.close()
                else:
                    with codecsopen("users.txt", 'w', encoding='utf-8') as f:
                        f.write(str(userId) + "\n")
                    f.close()
            else:
                create_message_box(
                    "Whoops!",
                    "Please enter a User ID to start a trial!", wx.OK)


# reads a given file and looks for a help tip attached to a given id
def display_help_pointers(filename, helpId):
    global mouseHelp
    if(ospath.isfile(filename)):
        with codecsopen(filename, 'r', encoding='utf-8') as f:
            try:
                file_data = jsonload(f)
                mouseHelp = file_data[helpId]["helpTip"]
                f.close()
            except ValueError:
                warn('File ' + str(filename) + ' was an invalid JSON file.')
    else:
        warn('File ' + str(filename) + ' could not be found.')


# tests wheter a set of arguments passed to a function are of the correct types
def test_values(inputVariables, valueArray, typeArray, functionCaller):
    for counter in range(0, len(valueArray)):
        if(isinstance(typeArray[counter], list)):
            possible_variable_types = typeArray[counter]
            passed = False
            for variableType in possible_variable_types:
                if(isinstance(
                   inputVariables.get(valueArray[counter]), variableType)):
                    passed = True
            if(passed is False):
                warn(
                    'Argument ' +
                    str(valueArray[counter]) +
                    ' passed to ' +
                    str(functionCaller) +
                    ' was of incorrect type' +
                    str(type(inputVariables.get(valueArray[counter]))) +
                    ' and should be of type ' + str(possible_variable_types))
                return False
        else:
            if(not isinstance(
               inputVariables.get(valueArray[counter]), typeArray[counter])):
                warn('Argument ' + str(valueArray[counter]) +
                     ' passed to ' +
                     str(functionCaller) +
                     ' was of incorrect type ' +
                     str(type(inputVariables.get(valueArray[counter]))) +
                     ' and should be of type ' + str(typeArray[counter]))
                return False
    return True


# adds a button to the array of buttons
def add_button(xpos, ypos, width, height,
               tcolor, bcolor, lcolor, caption,
               display_window, open_window=0, functionCall=0,
               functionArg=0, textSize=12, scrollBar=False,
               isTemp=False, prioritizeTask=False, fontName='Verdana'):
    global buttons
    should_add = test_values(locals(), getargspec(add_button)[0], [
        int, int, int, int, tuple, tuple, tuple, [str, unicode],
        int, int, [int, str], [int, list, tuple],
        int, [bool, int], bool, bool, str], 'add_button')
    if(should_add):
        return buttons.append((
            xpos, ypos, width, height,
            tcolor, bcolor, lcolor, caption,
            display_window, open_window, functionCall,
            functionArg, textSize, scrollBar, isTemp, prioritizeTask, fontName))


def add_window(newWindow):
    global windows
    should_add = test_values(
        locals(), getargspec(add_window)[0], [gui_fx.MenuWindow], 'add_window')
    if(should_add):
        return windows.append(newWindow)


def add_input(inputObject, xpos, ypos, width,
              height, windowid, textsize, scrollBar=False):
    global windows
    should_add = test_values(locals(), getargspec(
        add_input)[0],
        [gui_fx.InputField, int, int, int, int, int, int, int], 'add_input')
    if(should_add):
        return inputFields.append((
            inputObject, xpos, ypos, width, height, windowid, textsize, scrollBar))


def add_text(xpos, ypos, color, size, text, window, scrollBar=False):
    global textBoxes
    should_add = test_values(locals(), getargspec(add_text)[0], [
        int, int, tuple, int, [str, unicode], int, [bool, int]], 'add_text')
    if(should_add):
        return textBoxes.append((xpos, ypos, color,
                                 size, text, window, scrollBar))


def add_scroll(newBar):
    global scroll_bars
    should_add = test_values(locals(), getargspec(add_scroll)[0],
                             [tuple], 'add_scroll')
    if(should_add):
        newBar[0].baseContentHeight = newBar[0].contentHeight
        return scroll_bars.append(newBar)


def add_image(centerx, centery, filename, windowId, sizex, sizey, scrollBar):
    global images
    should_add = test_values(locals(), getargspec(add_image)[0], [[int, float],
                             [int, float], str, int, float, float,
                             [int, bool]], 'add_image')
    if(should_add):
        try:
            image = pyglet.image.load(filename)
            sprite = pyglet.sprite.Sprite(image, x=centerx, y=centery)
            sprite.scale = sizex / image.width
            return images.append((sprite, windowId, scrollBar,
                                  centery, centerx))
        except IOError:
            warn("File '" + filename + "' not found!")


def add_switch(newSwitch):
    global switches
    should_add = test_values(locals(), getargspec(add_switch)[0],
                             [tuple], 'add_switch')
    if(should_add):
        return switches.append(newSwitch)


# draws a button
def draw_button(centerx, centery, width, height,
                textColor, button_color, outlineColor,
                text, textSize, scroll_id, fontName='Verdana'):
    if(test_values(locals(), getargspec(draw_button)[0], [[float, int],
                   [float, int], [float, int], [float, int],
                   tuple, tuple, tuple,
                   [str, unicode], int, int, str], 'draw_button')):
        if(button_color[3] == 255):
            bottom_color = (button_color[0] - 15, button_color[1] - 15,
                            button_color[2] - 15, button_color[3]) \
                if ((button_color[0] - 15) >= 0 and
                    (button_color[1] - 15) >= 0 and
                    (button_color[2] - 15) >= 0) else button_color
            vertex_list = pyglet.graphics.vertex_list(
                4,
                ('v2f',
                    ((centerx - (width / 2)), (centery - (height / 2)),
                     (centerx + (width / 2)), (centery - (height / 2)),
                     (centerx + (width / 2)), (centery + (height / 2)),
                     (centerx - (width / 2)), (centery + (height / 2)))),
                ('c4B',
                    bottom_color + bottom_color + button_color + button_color))
            vertex_list.draw(pyglet.gl.GL_QUADS)
        vertex_list = pyglet.graphics.vertex_list(
            4,
            ('v2f',
                ((centerx - (width / 2)), (centery - (height / 2)),
                    (centerx + (width / 2)), (centery - (height / 2)),
                    (centerx + (width / 2)), (centery + (height / 2)),
                    (centerx - (width / 2)), (centery + (height / 2)))),
            ('c4B', outlineColor + outlineColor + outlineColor + outlineColor))
        vertex_list.draw(pyglet.gl.GL_LINE_LOOP)
        draw_text(centerx, centery, textColor, textSize, text,
                  scroll_id, width, fontName)
        return True
    else:
        warn("Button with caption " + str(text) + " was not drawn.")
        return False


# parameters: width of window, height of window, scrollBar class
def draw_bar(windowWidth, windowHeight, bar):
    if(test_values(locals(), getargspec(draw_bar)[0],
                   [int, int, gui_fx.ScrollBar], 'draw_bar')):
        vertex_list = pyglet.graphics.vertex_list(
            4,
            ('v2f', ((windowWidth), (windowHeight),
             (windowWidth - (windowWidth / 30)), (windowHeight),
             (windowWidth - (windowWidth / 30)), (0),
             (windowWidth), (0))),
            ('c4B',
                (173, 173, 173, 255, 173, 173, 173, 255, 173,
                 173, 173, 255, 173, 173, 173, 255))
        )
        vertex_list.draw(pyglet.gl.GL_QUADS)
        vertex_list = pyglet.graphics.vertex_list(
            4,
            ('v2f', ((windowWidth), (windowHeight - bar.yPos),
             (windowWidth - (windowWidth / 30)), (windowHeight - bar.yPos),
             (windowWidth - (windowWidth / 30)), (
                windowHeight - bar.yPos - scrollBarHeight),
             (windowWidth), (windowHeight - bar.yPos - scrollBarHeight))),
            ('c4B',
                (235, 235, 235, 255, 235, 235, 235, 255, 235,
                 235, 235, 255, 235, 235, 235, 255))
        )
        vertex_list.draw(pyglet.gl.GL_QUADS)
        return True
    else:
        warn("Scroll bar failed to draw")
        return False


# draws a switch, with the button color varying depending on how the switch
def draw_switch(windowWidth, windowHeight, switch, scroll_id):
    if(test_values(locals(), getargspec(draw_switch)[0],
                   [int, int, gui_fx.BooleanSwitch, int], 'draw_switch')):
        # tmp y position that is altered when switch is attached to a scrollbar
        centery = switch.centery
        centerx = switch.centerx
        if(scroll_id is not False):
            for eachScrollBar in scroll_bars:
                if(eachScrollBar[0].scroll_id == scroll_id):
                    if(eachScrollBar[0].isHorizontal):
                        centerx = centerx + eachScrollBar[0].relativeYPos
                    else:
                        centery = centery + eachScrollBar[0].relativeYPos
        if(centery <= main_window_height and centery >= 0):
            draw_button(
                switch.centerx, centery, switch.width, switch.height,
                (30, 28, 24, 255), (48, 51, 50, 255), (15, 15, 14, 255), '', 1,
                scroll_id
            )
            draw_button(
                switch.centerx - int(switch.width / 4), centery,
                int(switch.width * 0.45), int(
                    switch.height * 0.85), (30, 28, 24, 255)
                if switch.booleanValue
                else (242, 237, 208, 255), (102, 186, 50, 255)
                if switch.booleanValue
                else (27, 40, 11, 255), (15, 15, 14, 255),
                'True', switch.textSize, scroll_id, 'Arial')
            draw_button(
                switch.centerx + int(switch.width / 4), centery,
                int(switch.width * 0.45), int(switch.height * 0.85),
                (242, 237, 208, 255) if switch.booleanValue
                else (30, 28, 24, 255),
                (56, 13, 7, 255) if switch.booleanValue
                else (219, 30, 30, 255),
                (15, 15, 14, 255), 'False', switch.textSize, scroll_id, 'Arial'
            )
            return True
    else:
        warn("Switch failed to draw")
        return False


# draws text
def draw_text(centerx, centery, textColor, textSize, text,
              scroll_id=False, buttonWidth=0, fontName='Verdana'):
    global labelCache
    if(test_values(locals(), getargspec(draw_text)[0],
                   [[float, int], [float, int],
                   tuple, int, [unicode, str],
                   [int, bool], [int, float], str], 'draw_text')):
        label_name = text + str(textSize) + str(textColor)
        if (label_name) in labelCache:
            if((isinstance(scroll_id, bool))):
                labelCache[label_name].draw()
            else:
                labelCache[label_name].x = centerx
                labelCache[label_name].y = centery
                labelCache[label_name].anchor_x = 'center'
                labelCache[label_name].anchor_y = 'center'
                labelCache[label_name].draw()
        else:
            label = pyglet.text.Label(
                text, font_name=fontName, font_size=textSize, x=centerx,
                y=centery, anchor_x='center', anchor_y='center',
                color=textColor, multiline=True if buttonWidth != 0 else False,
                width=buttonWidth if buttonWidth != 0 else None, align='center'
            )
            labelCache[label_name] = label
            label.draw()
        return True
    else:
        warn("Text with caption " + text + " failed to draw")
        return False


# writes the values of all current input boxes to a file. Parameters: name of
# config section, list of names of values to be written
def writeValuesToFile(section_names, field_names, filename=None):
    try:
        dialog = wx.FileDialog(wxWindow, "Save Config As...",
                               executable, ".json", "", wx.FD_SAVE)
        if(filename is None):
            result = dialog.ShowModal()
            output_path = dialog.GetDirectory() + "\\" + dialog.Getfilename()
        else:
            result = None
            output_path = filename
        if (result == wx.ID_OK or filename is not None):
            objects_list = OrderedDict()
            for counter3 in range(0, len(section_names)):
                section_name = section_names[counter3]
                for counter in range(0, (len(field_names))):
                    d = OrderedDict()
                    for eachInputField in inputFields:
                        if(field_names[counter] == eachInputField[0].name and section_name == eachInputField[0].section_name):
                            d['value'] = eachInputField[0].text
                            objects_list[eachInputField[0].name] = d
                    for eachSwitch in switches:
                        if(eachSwitch[0].attachedValueName == field_names[counter] and section_name == eachSwitch[0].section_name):
                            d['value'] = 'true' if eachSwitch[0].booleanValue else 'false'
                            objects_list[eachSwitch[0].attachedValueName] = d
            if(ospath.isfile(output_path)):
                output = open(str(output_path), 'w')
            else:
                output = open(str(output_path), 'a')
            j = jsondumps(objects_list, indent=2)
            output.write(j)
            output.close()
            dialog.Destroy()
    except TypeError:
        warn('Failed to create dialog')


# reads the values of all current input boxes from a file, then changes the input box text accordingly. Parameters: name of config section, list of names of values to be written.
def read_values_from_file(section_names, field_names, filename=None):
    global inputFields
    global switches
    try:
        dialog = wx.FileDialog(wxWindow, "Select Config File", executable, ".json", "", wx.FD_OPEN)
        if(filename == None):
            result = dialog.ShowModal()
            readpath = dialog.GetDirectory() + "\\" + dialog.Getfilename()
        else:
            result = None
            readpath = filename
        if (result == wx.ID_OK or filename != None):
            if(ospath.isfile(readpath)):
                with codecsopen(str(readpath), 'r', encoding='utf-8') as f:
                    try:
                        file_data = jsonload(f)
                        for counter3 in range(0, len(section_names)):
                            section_name = section_names[counter3]
                            for counter in range(0, (len(field_names))):
                                for counter2 in range(0, len(inputFields)):
                                    if(counter != len(field_names)):
                                        if(field_names[counter] == inputFields[counter2][0].name and section_name == inputFields[counter2][0].section_name):
                                            inputFields[counter2][0].text = file_data[inputFields[counter2][0].name]["value"]
                                            counter = len(field_names)
                                for counter2 in range(0, len(switches)):
                                    if(counter != len(field_names)):
                                        if(switches[counter2][0].attachedValueName == field_names[counter] and section_name == switches[counter2][0].section_name):
                                            switches[counter2][0].booleanValue = (True if (file_data[(field_names[counter])]["value"]) == 'true' else False)
                                            counter = len(field_names)
                    except ValueError:
                        warn('File ' + str(readpath) + " is an invalid JSON file.")
                    except KeyError:
                        warn('File ' + str(readpath) + " does not contain all parameter values.")
                f.close()
            else:
                warn("File " + str(readpath) + " could not be found.")
        dialog.Destroy()
    except TypeError:
        warn('Failed to create dialog')


# Creates a wxpython message box with a given title, text, and type.
# Currently does not do anything based on which button the user clicks.
def create_message_box(title, text, thetype):
    try:
        dialog = wx.MessageDialog(wxWindow, text, title, thetype)
        result = dialog.ShowModal()
        if result == wx.ID_YES:
            pass
        dialog.Destroy()
        return True
    except TypeError:
        warn('Failed to create dialog with arguments: title: ' + title + " text: " + text + " type: " + str(thetype))


# runs a given python file
def run_python_file(filename):
    if(ospath.isfile(filename)):
        try:
            execfile(filename)
            return True
        except SyntaxError:
            warn("File " + str(filename) + " is not a valid Python file.")
    else:
        warn("File " + str(filename) + " not found.")


# runs BCI main to start the experiment
def exec_bci_main(parameters, window, mode):

    # Get experiment information from globally set variables
    global trialType
    global userId
    global new_window

    # set textBoxes, buttons, inputs, and images globally
    global textBoxes
    global buttons
    global inputFields
    global images

    try:

        new_window.close()
        bci_main.bci_main(parameters, userId, trialType, mode)

    except Exception as e:
        if e.message == 'Not implemented yet!':
            message = "BCI mode not Implemented yet!"

        elif e.strerror == 'Address already in use':
            message = "Please close all BCI Windows and restart!"

        else:
            message = "Error in BCI Main Execution: %s" % (e)

        create_message_box(
            "BCI Error",
            message,
            wx.OK)


# Runs a command (filename) from the given location (execPath). Intended to
# run exectuable files. If isParameter is true, filename should be the name of the
# parameter containing the name of the exe file.
def run_executable(execPath, filename, isParameter):
    if(isParameter):
        for eachInputField in inputFields:
            if(filename == eachInputField[0].name):
                filename = eachInputField[0].text
    try:
        dir_path = ospath.dirname(ospath.realpath(__file__))
        chdir(execPath)
        subprocess.call(filename)
        chdir(ospath.dirname(dir_path))
    except Exception:
        warn("Could not find executable " + filename + " in path " + execPath)
        chdir(ospath.dirname(dir_path))


# removes drop-down items that are on the screen. Called when the user clicks
# anywhere after opening a drop-down list.
def remove_dropdown_list():
    buttonTrueIndex = []
    for counter in range(0, (len(buttons))):
        if buttons[counter][14] is True:
            buttonTrueIndex.append(buttons[counter])
    for counter2 in range(0, (len(scroll_bars))):
        for counter in range(0, len(buttonTrueIndex)):
            if(scroll_bars[counter2][0].scroll_id == buttonTrueIndex[counter][13]):
                if(scroll_bars[counter2][0].baseContentHeight == 0):
                    scroll_bars[counter2][0].yPos = 0
                scroll_bars[counter2][0].resetContentHeight()
                scroll_bars[counter2][0].translateBarToMovement(scroll_bars[counter2][0].yPos)
                try:
                    buttons.remove(buttonTrueIndex[counter])
                except:
                    pass
    buttonTrueIndex = []


class BooleanSwitch():
    def __init__(self, par2xpos, par3ypos,
                 par4width, par5height, par6name,
                 par7defaultvalue, par8textsize, par9section_name):
        self.centerx = par2xpos
        self.centery = par3ypos
        self.width = par4width
        self.height = par5height
        self.attachedValueName = par6name
        self.booleanValue = par7defaultvalue
        self.textSize = par8textsize
        self.section_name = par9section_name


# input box for the user to type in
class InputField():
    def __init__(self, par2Name, par3section_name, par4IsNumeric):
        self.text = ""
        self.name = par2Name
        self.section_name = par3section_name
        self.isNumeric = par4IsNumeric


class ScrollBar():
    def __init__(self, par2WindowHeight, theID, relativeYPos = 0, visible=True, horizontal=False):
        # current position of the scroller
        self.yPos = 0
        # position of the content being scrolled
        self.relativeYPos = relativeYPos
        # actual height of the bar
        self.height = par2WindowHeight
        # height of the content being scrolled
        self.contentHeight = 0
        # is the scroll bar visible- used for scrolling drop-down lists
        self.isVisible = visible
        # the height of any content before drop-down menus are addedHeight
        self.baseContentHeight = 0
        # does the bar scroll horizontally?
        self.isHorizontal = horizontal
        # the id of this bar
        self.scroll_id = theID

    def addToContentHeight(self, addedHeight):
        self.contentHeight = self.contentHeight + addedHeight

    def resetContentHeight(self):
        self.contentHeight = self.baseContentHeight

    #calculates how much the attached content needs to be moved based on the amount the bar has been scrolled
    def translateBarToMovement(self, bar_amount):
        #a/b = c/d ad = bc
        #bar_amount/height = x/contentHeight bar_amount*contentHeight = height*x x=(bar_amount*contentHeight)/height
        self.relativeYPos = (bar_amount*(self.contentHeight/(640.0 if self.isHorizontal else 480.0))*(main_window_width if self.isHorizontal else main_window_height))/(self.height - scrollBarHeight)
        return self.relativeYPos

#window that opens when a button is pressed. window id is used to determine what should display in the window
class MenuWindow(pyglet.window.Window):

    def __init__(self, par2WindowId, par3Title):
        global main_window_width
        global main_window_height
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        self.windowId = par2WindowId
        main_window_width = int((screen.width/3)*2)
        main_window_height = int((screen.height/4)*3)
        super(MenuWindow, self).__init__(caption=(par3Title if(par3Title != ' ') else ("Window " + str(par2WindowId))), width=int((screen.width/3)*2), height=int((screen.height/4)*3))

    # draws all input fields
    def draw_input_fields(self):
        global currentActiveInputField
        returnTrue = True
        for counter in range(0, (len(inputFields))):
            if(counter == currentActiveInputField):
                mainbutton_color = (207, 207, 207, 255)
                text = inputFields[counter][0].text[:typingCursorPos] + "|" + inputFields[counter][0].text[typingCursorPos:]
            else:
                mainbutton_color = (247, 247, 247, 255)
                text = inputFields[counter][0].text
            if(inputFields[counter][5] == self.windowId):
                scroll_id = inputFields[counter][7]
                centery = inputFields[counter][2]
                centerx = inputFields[counter][1]
                if(scroll_id != False):
                    for eachScrollBar in scroll_bars:
                        if(eachScrollBar[0].scroll_id == scroll_id):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                tempWidth = inputFields[counter][3]
                if(tempWidth < len(text) * inputFields[counter][6] * 0.7):
                    tempWidth = len(text) * inputFields[counter][6] * 0.7
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    if(not draw_button(centerx, centery, tempWidth, inputFields[counter][4], (0, 0, 0, 255), mainbutton_color, (117, 117, 117, 255), text, inputFields[counter][6], scroll_id, 'Arial')):
                        returnTrue = False
        return returnTrue

    # draws all text
    def draw_text_boxes(self):
        returnTrue = True
        for counter in range(0, (len(textBoxes))):
            if(textBoxes[counter][5] == self.windowId):
                scroll_id = textBoxes[counter][6]
                centery = textBoxes[counter][1]
                centerx = textBoxes[counter][0]
                if(scroll_id != False):
                    for eachScrollBar in scroll_bars:
                        if(eachScrollBar[0].scroll_id == scroll_id):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    if(not draw_text(centerx, centery, textBoxes[counter][2], textBoxes[counter][3], textBoxes[counter][4], textBoxes[counter][5])):
                        returnTrue = False
        return returnTrue

    # draws all buttons
    def draw_buttons(self):
        global buttonsOnScreen
        returnTrue = True
        buttonsOnScreen = []
        for counter in range(0, (len(buttons))):
            if(mouseOnButton == counter):
                mainbutton_color = (max(0, buttons[counter][5][0] - 40), max(0, buttons[counter][5][1] - 40), max(0, buttons[counter][5][2] - 40), buttons[counter][5][3])
            else:
                mainbutton_color = buttons[counter][5]
            if(buttons[counter][8] == self.windowId):
                scroll_id = buttons[counter][13]
                centery = buttons[counter][1]
                centerx = buttons[counter][0]
                if(not(isinstance(scroll_id, bool))):
                    for eachScrollBar in scroll_bars:
                        if(eachScrollBar[0].scroll_id == scroll_id):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    buttonsOnScreen.append(counter)
                    if(not draw_button(centerx, centery, buttons[counter][2], buttons[counter][3], buttons[counter][4], mainbutton_color, buttons[counter][6], buttons[counter][7], buttons[counter][12], scroll_id, buttons[counter][16])):
                        returnTrue = False
        return returnTrue

    # draws scroll bars
    def draw_scroll_bars(self):
        for counter in range(0, (len(scroll_bars))):
            if(scroll_bars[counter][0].scroll_id == self.windowId and scroll_bars[counter][0].isVisible == True):
                return draw_bar(self.width, self.height, scroll_bars[counter][0])

    # draws images
    def draw_images(self):
        global images
        returnTrue = True
        for counter in range(0, (len(images))):
            if(images[counter][1] == self.windowId):
                scroll_id = images[counter][2]
                centery = images[counter][3]
                centerx = images[counter][4]
                if(not(isinstance(scroll_id, bool))):
                    for eachScrollBar in scroll_bars:
                        if(eachScrollBar[0].scroll_id == scroll_id):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    images[counter][0].y = centery
                    images[counter][0].x = centerx
                    try:
                        images[counter][0].draw()
                    except:
                        returnTrue = False
        return returnTrue

    # draws switches
    def draw_switches(self):
        returnTrue = True
        for counter in range(0, (len(switches))):
            if(switches[counter][1] == self.windowId):
                if(not draw_switch(main_window_width, main_window_height, switches[counter][0], switches[counter][2])):
                    returnTrue = False
        return returnTrue

    # draws buttons, inputs, etc.
    def on_draw(self):
        global mouseX
        global mouseY
        returnTrue = True

        super(MenuWindow, self).clear()

        #background
        draw_button(main_window_width / 2, main_window_height / 2, main_window_width, main_window_height, (16, 19, 22, 255), (16, 19, 22, 255), (16, 19, 22, 255), '', 0, False)

        if(not self.draw_input_fields()):
            returnTrue = False
        if(not self.draw_text_boxes()):
            returnTrue = False
        if(not self.draw_switches()):
            returnTrue = False
        if(not self.draw_buttons()):
            returnTrue = False
        if(not self.draw_scroll_bars()):
            returnTrue = False
        if(not self.draw_images()):
            returnTrue = False

        #draws a help tip box at the location of the mouse pointer if mouseHelp is set
        if(mouseHelp != False):
            if(not draw_button(int(mouseX - (50/640.0)*main_window_width), int(mouseY - ((10*(len(mouseHelp)/10.0 + 1) + 15)/960.0)*main_window_height), int((100/640.0)*main_window_width), int(((10*(len(mouseHelp)/10.0 + 1) + 15)/480.0)*main_window_height), (11, 4, 22, 255), (224, 217, 204, 255), (56, 55, 58, 255), mouseHelp, int((9/640.0)*main_window_width), 0, fontName='Arial')):
                returnTrue = False
        return returnTrue

    # determines which, if any, button the mouse is over
    def on_mouse_motion(self, x, y, dx, dy):
        global mouseOnButton
        global mouseHelp
        global mouseX
        global mouseY

        changed_mouse = False
        changed_input = False
        mouseX = x
        mouseY = y

        # Checks whethter the mouse is over a button, and, if so, changes the relevant variable
        for counter in range(0, (len(buttonsOnScreen))):
            button_index = buttonsOnScreen[counter]
            if(button_index < len(buttons)):
                if(self.windowId == buttons[button_index][8]):
                    temp_height = buttons[button_index][1]
                    centerx = buttons[button_index][0]
                    if(not(isinstance(buttons[button_index][13], bool))):
                        for eachScrollBar in scroll_bars:
                            if(eachScrollBar[0].scroll_id == buttons[button_index][13]):
                                if(eachScrollBar[0].isHorizontal):
                                    centerx = centerx + eachScrollBar[0].relativeYPos
                                else:
                                    temp_height = temp_height + eachScrollBar[0].relativeYPos
                    if((x <= (centerx + buttons[button_index][2]/2)) and (x >= (centerx - buttons[button_index][2]/2))):
                        if((y <= (temp_height + buttons[button_index][3]/2)) and (y >= (temp_height - buttons[button_index][3]/2))):
                            mouseOnButton = button_index
                            changed_mouse = True
                            counter = len(buttonsOnScreen)
        if(changed_mouse == False):
            mouseOnButton = 1000
            mouseHelp = False

    # scroll bar handling
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        for counter in range(0, (len(scroll_bars))):
            if(scroll_bars[counter][0].scroll_id == self.windowId):
                if(y < self.height and y > 0):
                    if((x <= self.width) and (x >= (self.width - (self.width / 30)))):
                        if((y <= self.height - scroll_bars[counter][0].yPos + scrollBarHeight) and (y >= self.height - scroll_bars[counter][0].yPos - scrollBarHeight)):
                            scroll_bars[counter][0].yPos = self.height - y
                            if(scroll_bars[counter][0].yPos < 0):
                                scroll_bars[counter][0].yPos = 0
                            if(scroll_bars[counter][0].yPos > self.height - scrollBarHeight):
                                scroll_bars[counter][0].yPos = self.height - scrollBarHeight
                            scroll_bars[counter][0].translateBarToMovement(self.height - y)
                return

    # more scroll bar handling
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        for counter in range(0, (len(scroll_bars))):
            if(scroll_bars[counter][0].scroll_id == self.windowId):
                scroll_bars[counter][0].yPos = scroll_bars[counter][0].yPos - (scroll_y * 3)
                if(scroll_bars[counter][0].yPos < 0):
                    scroll_bars[counter][0].yPos = 0
                if(scroll_bars[counter][0].yPos > self.height - scrollBarHeight):
                    scroll_bars[counter][0].yPos = self.height - scrollBarHeight
                scroll_bars[counter][0].translateBarToMovement(scroll_bars[counter][0].yPos)
                return

    # opens the window attached to a button, or executes the function associated with the button
    def on_mouse_press(self, x, y, button, modifiers):
        global scroll_bars
        global currentActiveInputField
        global switches
        global mouseOnSwitch
        global mouseOnInput
        global typingCursorPos

        # make sure that drop-down items can run their task before they are deleted
        if(mouseOnButton != 1000 and mouseOnButton < len(buttons)):
            if(buttons[mouseOnButton][15] == True):
                if(buttons[mouseOnButton][10] != 0):
                    getattr(gui_fx, buttons[mouseOnButton][10])(*(buttons[mouseOnButton][11]))

        remove_dropdown_list()

        if(currentActiveInputField != 1000):
            if(inputFields[currentActiveInputField][0].isNumeric == True):
                try:
                    float(inputFields[currentActiveInputField][0].text)
                except ValueError:
                    create_message_box("Warning", "The parameter " + inputFields[currentActiveInputField][0].name + " takes a numeric value as input.", wx.ICON_EXCLAMATION)
            currentActiveInputField = 1000

        # run mouse tasks if necessary (window opening, function activation)
        if(mouseOnButton != 1000):
            if(mouseOnButton < len(buttons)):
                open_window = True
                for counter in range(0, (len(windows))):
                    if(windows[counter].windowId == buttons[mouseOnButton][9]):
                        open_window = False
                if(open_window == True):
                    new_window = MenuWindow(buttons[mouseOnButton][9], ' ')
                    add_window(new_window)
                if(buttons[mouseOnButton][10] != 0):
                    getattr(gui_fx, buttons[mouseOnButton][10])(*(buttons[mouseOnButton][11]))
        else:

            # detects whether a text box has been clicked
            changed_input = False
            for counter in range(0, (len(inputFields))):
                if(self.windowId == inputFields[counter][5]):
                    temp_height = inputFields[counter][2]
                    centerx = inputFields[counter][1]
                    if(not(isinstance(inputFields[counter][7], bool))):
                        for eachScrollBar in scroll_bars:
                            if(eachScrollBar[0].scroll_id == inputFields[counter][7]):
                                if(eachScrollBar[0].isHorizontal):
                                    centerx = centerx + eachScrollBar[0].relativeYPos
                                else:
                                    temp_height = temp_height + eachScrollBar[0].relativeYPos
                    if((x <= (centerx + inputFields[counter][3]/2)) and (x >= (centerx - inputFields[counter][3]/2))):
                        if((y <= (temp_height + inputFields[counter][4]/2)) and (y >= (temp_height - inputFields[counter][4]/2))):
                            mouseOnInput = counter
                            changed_input = True
                            typingCursorPos = len(inputFields[counter][0].text)
                            counter = len(inputFields)
            if(changed_input == False):
                mouseOnInput = 1000
            if(mouseOnInput != 1000):
                currentActiveInputField = mouseOnInput
            else:

                # detects whether a switch has been clicked
                changed_input = False
                for counter in range(0, (len(switches))):
                    if(self.windowId == switches[counter][1]):
                        temp_height = switches[counter][0].centery
                        centerx = switches[counter][0].centerx
                        if(not(isinstance(switches[counter][2], bool))):
                            for eachScrollBar in scroll_bars:
                                if(eachScrollBar[0].scroll_id == switches[counter][2]):
                                    if(eachScrollBar[0].isHorizontal):
                                        centerx = centerx + eachScrollBar[0].relativeYPos
                                    else:
                                        temp_height = temp_height + eachScrollBar[0].relativeYPos
                        if((x <= (centerx + switches[counter][0].width/2)) and (x >= (centerx - switches[counter][0].width/2))):
                            if((y <= (temp_height + switches[counter][0].height/2)) and (y >= (temp_height - switches[counter][0].height/2))):
                                mouseOnSwitch = counter
                                changed_input = True
                                counter = len(switches)
                if(changed_input == False):
                    mouseOnSwitch = 1000
                if(mouseOnSwitch != 1000):
                    switches[mouseOnSwitch][0].booleanValue = not(switches[mouseOnSwitch][0].booleanValue)

    # typing in input boxes
    def on_key_press(self, symbol, modifiers):
        global shiftPressed
        global currentActiveInputField
        global typingCursorPos

        # input field typing handling
        if(currentActiveInputField != 1000):
            try:
                if(symbol == pyglet.window.key.BACKSPACE):
                    if(len(inputFields[currentActiveInputField][0].text) > 0):
                        if(typingCursorPos > 0):
                            inputFields[currentActiveInputField][0].text = inputFields[currentActiveInputField][0].text[:typingCursorPos - 1] + inputFields[currentActiveInputField][0].text[typingCursorPos:]
                            typingCursorPos = typingCursorPos - 1
                elif(symbol == pyglet.window.key.MOTION_LEFT):
                    if(typingCursorPos != 0):
                        typingCursorPos = typingCursorPos - 1
                elif(symbol == pyglet.window.key.MOTION_RIGHT):
                    if(typingCursorPos < len(inputFields[currentActiveInputField][0].text)):
                        typingCursorPos = typingCursorPos + 1
                elif(len(pyglet.window.key.symbol_string(symbol)) == 1):
                    if(shiftPressed):
                        insert_symbol_at_index(pyglet.window.key.symbol_string(symbol), currentActiveInputField)
                    else:
                        insert_symbol_at_index(str.lower(pyglet.window.key.symbol_string(symbol)), currentActiveInputField)
                elif(str.isdigit(pyglet.window.key.symbol_string(symbol)[1])):
                    insert_symbol_at_index(pyglet.window.key.symbol_string(symbol)[1], currentActiveInputField)
                elif(symbol == pyglet.window.key.LSHIFT or symbol == pyglet.window.key.RSHIFT):
                    shiftPressed = True
                elif(symbol == pyglet.window.key.SPACE):
                    insert_symbol_at_index(" ", currentActiveInputField)
                elif(symbol == pyglet.window.key.TAB):
                    for counter2 in range(0, (len(inputFields))):
                        if(counter2 != currentActiveInputField and inputFields[counter2][2] <= inputFields[currentActiveInputField][2] and inputFields[currentActiveInputField][5] == inputFields[counter2][5]):
                            currentActiveInputField = counter2
                            return
                    currentHighestField = currentActiveInputField
                    for counter2 in range(0, (len(inputFields))):
                        if(counter2 != currentActiveInputField and inputFields[counter2][2] >= inputFields[currentHighestField][2] and inputFields[currentActiveInputField][5] == inputFields[counter2][5]):
                            currentHighestField = counter2
                    currentActiveInputField = currentHighestField
                elif(str.isdigit(pyglet.window.key.symbol_string(symbol)[4])):
                        insert_symbol_at_index(pyglet.window.key.symbol_string(symbol)[4], currentActiveInputField)
                elif(symbol == pyglet.window.key.MINUS):
                    if(shiftPressed):
                        insert_symbol_at_index("_", currentActiveInputField)
                    else:
                        insert_symbol_at_index("-", currentActiveInputField)
                elif(symbol == pyglet.window.key.PERIOD):
                        insert_symbol_at_index(".", currentActiveInputField)
                elif(symbol == pyglet.window.key.CAPSLOCK):
                    shiftPressed = (not shiftPressed)
                elif(symbol == pyglet.window.key.APOSTROPHE):
                        insert_symbol_at_index("'", currentActiveInputField)
                elif(symbol == pyglet.window.key.BRACKETLEFT):
                    if(shiftPressed):
                        insert_symbol_at_index("{", currentActiveInputField)
                    else:
                        insert_symbol_at_index("[", currentActiveInputField)
                elif(symbol == pyglet.window.key.BRACKETRIGHT):
                    if(shiftPressed):
                        insert_symbol_at_index("}", currentActiveInputField)
                    else:
                        insert_symbol_at_index("]", currentActiveInputField)
            except IndexError:
                print "Invalid key press"

    # for detecting shift key usage for input box text
    def on_key_release(self, symbol, modifiers):
        global shiftPressed
        if(symbol == pyglet.window.key.LSHIFT or symbol == pyglet.window.key.RSHIFT):
            shiftPressed = False

    # removes window from window array so that it can be opened again, or, if this is the main window, closes all windows
    def on_close(self):
        for counter in range(0, (len(scroll_bars))):
            if(scroll_bars[counter][0].scroll_id == self.windowId):
                scroll_bars[counter][0].yPos = 0
                scroll_bars[counter][0].translateBarToMovement(0)
        if(self.windowId == 0):
            for counter in range(0, len(windows)):
                if(windows[counter].windowId != self.windowId):
                    windows[counter].close()
            wxWindow.Close()
            super(MenuWindow, self).close()
            return

        # close advanced parameters window if main parameters window is closed
        elif(self.windowId == 3):
            for counter in range(0, len(windows)):
                if(windows[counter].windowId == 4):
                    windows[counter].close()
                    del windows[counter]
        for counter in range(0, len(windows)):
            if(windows[counter].windowId == self.windowId):
                del windows[counter]
                super(MenuWindow, self).on_close()
                return
        super(MenuWindow, self).on_close()
