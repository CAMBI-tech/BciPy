import codecs
import os
import pyglet
import pytest
import sys
sys.path.append('gui/utility')
import gui_fx
import wx
from warnings import simplefilter

simplefilter = simplefilter('ignore')


def teardown_function(function):
    gui_fx.inputFields = []
    gui_fx.scroll_bars = []
    gui_fx.buttons = []
    gui_fx.switches = []
    gui_fx.textBoxes = []
    gui_fx.images = []
    gui_fx.mouseOnInput = 1000
    gui_fx.mouseOnButton = 1000
    gui_fx.mouseOnSwitch = 1000
    gui_fx.mouseHelp = False
    gui_fx.currentActiveInputField = 1000
    gui_fx.shiftPressed = True
    gui_fx.windows = []
    gui_fx.buttonsOnScreen = []
    gui_fx.main_window_height = 480
    gui_fx.main_window_width = 640


def test_changeInputText():
    '''Tests that the function changeInputText can change the text of an input field'''
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 100, 100, 100, 0, 16
    )
    gui_fx.changeInputText('test_id', "some_test_text")
    assert gui_fx.inputFields[0][0].text == "some_test_text"


def test_dropItems():
    '''Tests that the dropItems function can create a drop-down list of buttons based on values in a text or json file, and that those items can be removed'''
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 100, 100, 100, 0, 16
    )
    output = open(str('test.txt'), 'w')
    #test standard text file
    output.write('1\n2\n3\n4\n5\n6\n')
    output.close()
    gui_fx.dropItems('test_id', 0, 'test.txt', False)
    assert len(gui_fx.buttons) == 6
    #test json file
    gui_fx.buttons = []
    output = open(str('test.txt'), 'w')
    output.write('{"test_id":{"test_values": ["1", "2", "3"]}}')
    output.close()
    gui_fx.dropItems('test_id', 0, 'test.txt', "test_values")
    os.remove('test.txt')
    assert len(gui_fx.buttons) == 3
    gui_fx.remove_dropdown_list()
    #test whether remove_dropdown_list removes items
    assert len(gui_fx.buttons) == 0


def test_setTrialType():
    '''Tests that the setTrialType function can add a newly entered user to the users.txt file, and that it will not add that user more than once'''
    gui_fx.add_input(
        gui_fx.InputField('user_id', False, False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.inputFields[0][0].text = "test"
    tempData = ""
    if(os.path.isfile('users.txt')):
        output = open(str('users.txt'), 'r')
        tempData = output.read()
        output.close()
    if(os.path.isfile('users.txt')):
        output = open(str('users.txt'), 'w')
    else:
        output = open(str('users.txt'), 'a')
    output.write("")
    output.close()
    gui_fx.setTrialType(6)
    output = open(str('users.txt'), 'r')
    newData = output.read()
    output.close()
    #tests whether the data was written
    assert newData == "test\n"
    gui_fx.setTrialType(6)
    output = open(str('users.txt'), 'r')
    newData = output.read()
    output.close()
    #checks to make sure the data is not written if it already exists
    assert newData == "test\n"
    output = open(str('users.txt'), 'w')
    output.write(tempData)
    output.close()
    os.remove('users.txt')

def test_displayHelpPointers():
    '''Tests that the displayHelpPointers function can read a help tip from a json file and display it'''
    output = open(str('test.txt'), 'w')
    output.write('{"test_id":{"helpTip":"thisisahelptip"}}')
    output.close()
    gui_fx.displayHelpPointers("test.txt", "test_id")
    os.remove("test.txt")
    assert gui_fx.mouseHelp == "thisisahelptip"

def test_displayHelpPointersShouldFail():
    '''Tests various fail cases for displaying help pointers'''
    gui_fx.displayHelpPointers("invalid.txt", "test_id")
    gui_fx.displayHelpPointers("pytestfile.py", "test_id")
    assert gui_fx.mouseHelp == False

def test_writeValuesToFile():
    '''Tests that the writeValuesToFile function writes a correctly formatted string from input box values'''
    gui_fx.add_input(
        gui_fx.InputField('testInputOne', 'test_section_1', False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.add_input(
        gui_fx.InputField('testInputTwo', 'test_section_2', False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.changeInputText('testInputOne', 'testone')
    gui_fx.changeInputText('testInputTwo', 'testtwo')
    gui_fx.writeValuesToFile(['test_section_1', 'test_section_2'], ['testInputOne', 'testInputTwo'], "test.txt")
    output = open(str('test.txt'), 'r')
    tempData = output.read()
    output.close()
    assert tempData == '{\n  "testInputOne": {\n    "value": "testone"\n  }, \n  "testInputTwo": {\n    "value": "testtwo"\n  }\n}'
    os.remove("test.txt")

def test_read_values_from_file():
    '''Tests that the read_values_from_file function readds the correct values from a json file and writes them to the correct input boxes'''
    gui_fx.add_input(
        gui_fx.InputField('testInputOne', 'test_section_1', False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.add_input(
        gui_fx.InputField('testInputTwo', 'test_section_2', False), 100, 100, 100, 100, 0, 100
    )
    output = open(str('test.txt'), 'w')
    output.write('{\n  "testInputOne": {\n    "value": "testone"\n  }, \n  "testInputTwo": {\n    "value": "testtwo"\n  }\n}')
    output.close()
    gui_fx.read_values_from_file(['test_section_1', 'test_section_2'], ['testInputOne', 'testInputTwo'], "test.txt")
    assert (gui_fx.inputFields[0][0].text == 'testone' and gui_fx.inputFields[1][0].text == 'testtwo')
    os.remove("test.txt")

def test_onMouseMotion():
    '''Tests that the on_mouse_motion function correctly determines whether the mouse cursor is over a button'''
    gui_fx.addButton(
        1, 1, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    gui_fx.mouseX = 50
    gui_fx.mouseY = 50
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.draw_buttons()
    main_window.on_mouse_motion(50, 50, 0, 0)
    assert gui_fx.mouseOnButton == 0
    main_window.close()

def test_onMouseDrag():
    '''Tests that a scroll bar moves the appropriate amount after being dragged'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_scroll((gui_fx.ScrollBar(main_window.height, 0), 0))
    gui_fx.scroll_bars[0][0].addToContentHeight(10)
    main_window.on_mouse_drag(main_window.width - 10, main_window.height - 10, 0, 0, 0, 0)
    assert gui_fx.scroll_bars[0][0].relativeYPos == ((main_window.width - (main_window.width - 10))*(10/480.0)*main_window.height)/(main_window.height - gui_fx.scrollBarHeight)
    main_window.close()

def test_onMouseScroll():
    '''Tests that a scroll bar moves the appropriate amount relative to the mouse wheel'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_scroll((gui_fx.ScrollBar(main_window.height, 0), 0))
    gui_fx.scroll_bars[0][0].addToContentHeight(10)
    main_window.on_mouse_scroll(100, 100, 0, -10)
    assert gui_fx.scroll_bars[0][0].relativeYPos == (30*(10/480.0)*main_window.height)/(main_window.height - gui_fx.scrollBarHeight)
    main_window.close()

def test_onMousePress():
    '''Tests that the on_mouse_press function correctly determines whether the mouse cursor is over an input field or switch'''
    gui_fx.add_input(
        gui_fx.InputField('user_id', False, False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.add_switch(
        (gui_fx.BooleanSwitch(-100, -100, 100, 100, 'test', 'false', 100, 'test'), 0, 0)
    )
    gui_fx.mouseX = 50
    gui_fx.mouseY = 50
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.on_mouse_press(50, 50, 0, 0)
    #test input fields
    assert gui_fx.mouseOnInput == 0
    gui_fx.mouseX = -70
    gui_fx.mouseY = -70
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.on_mouse_press(-70, -70, 0, 0)
    #test switches
    assert gui_fx.mouseOnSwitch == 0
    main_window.close()

def test_onKeyPress():
    '''Tests that a key press when an input field is active appends the correct letter to the field'''
    gui_fx.add_input(
        gui_fx.InputField('user_id', False, False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.currentActiveInputField = 0
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.on_key_press(pyglet.window.key.A, None)
    main_window.close()
    assert gui_fx.inputFields[0][0].text == 'A'

def test_onKeyRelease():
    '''Tests shift key release'''
    gui_fx.add_input(
        gui_fx.InputField('user_id', False, False), 100, 100, 100, 100, 0, 100
    )
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.on_key_release(pyglet.window.key.LSHIFT, None)
    assert gui_fx.shiftPressed == False
    main_window.close()

def test_onClose():
    '''Tests window removal'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.close()
    assert len(gui_fx.windows) == 0
    main_window.close()

def test_addButton():
    '''Tests whether a button with correct parameters can be added'''
    gui_fx.addButton(
        0, 0, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    assert len(gui_fx.buttons) == 1

def test_addButtonShouldFail():
    '''Tests whether a button with incorrect parameters can be added. This should throw a warning.'''
    gui_fx.addButton(
        'test', 'test', 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    assert len(gui_fx.buttons) == 0

def test_add_input():
    '''Tests whether an input field with correct parameters can be added'''
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 100, 100, 100, 0, 16
    )
    assert len(gui_fx.inputFields) == 1

def test_add_inputShouldFail():
    '''Tests wheter an input field with incorrect parameters can be aded. This should throw a warning.'''
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 'test', False, 100, 100, 0, 16
    )
    assert len(gui_fx.inputFields) == 0

def test_add_text():
    '''Tests whether a text item with correct parameters can be added.'''
    gui_fx.add_text(
        100, 100, (247, 247, 247, 255), 100, "Parameters", 3, 3
    )
    assert len(gui_fx.textBoxes) == 1

def test_add_textShouldFail():
    '''Tests whether a text item with incorrect parameters can be added. This should throw a warning.'''
    gui_fx.add_text(
        100, 'test', (247, 247, 247, 255), 100, False, 3, 3
    )
    assert len(gui_fx.textBoxes) == 0

def test_add_window():
    '''Tests whether a valid MenuWindow can be added'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    assert len(gui_fx.windows) == 1
    main_window.close()

def test_add_scroll():
    '''Tests whether a valid ScrollBar can be added'''
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    assert len(gui_fx.scroll_bars) == 1

def test_add_image():
    '''Tests whether a valid image can be added'''
    gui_fx.add_image(
        100, 100, "static/images/OHSU-RGB-4C-REV.png", 0,
        float(39), float(67), False
    )
    assert len(gui_fx.images) == 1

def test_addNonexistentImage():
    '''Tests whether an IOError is caught if an image is nonexistent. This should throw a ing.'''
    gui_fx.add_image(
        100, 100, "test.png", 0,
        float(39), float(67), False
    )
    assert len(gui_fx.images) == 0

def test_add_switch():
    '''Tests whether a valid switch can be added'''
    gui_fx.add_switch(
        (gui_fx.BooleanSwitch(100, 100, 100, 100, 'test', True, 19, 'test'), 0, 0)
    )

def test_draw_buttons():
    '''Tests whether buttons on screen are drawn'''
    gui_fx.addButton(
        1, 1, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    gui_fx.addButton(
        1, 1, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    gui_fx.addButton(
        1, 1, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.draw_buttons()
    assert len(gui_fx.buttonsOnScreen) == 3
    main_window.close()

def test_run_python_fileShouldFail():
    '''Tests whether an attempt is made to run an invalid python file from the run_python_file function'''
    assert gui_fx.run_python_file("../users.txt") == None

def test_run_python_file():
    '''Tests whether a valid python file can be run from the run_python_file function'''
    assert gui_fx.run_python_file("gui/tests/testfile.py") == True

def test_run_executableShouldFail():
    '''Tests whether an attempt is made to run an invalid executable from the run_executable function'''
    assert gui_fx.run_executable('\\test', 'test.exe', False) == None

def test_create_message_boxShouldFail():
    '''Tests whether a message box can be created with invalid arguments'''
    assert gui_fx.create_message_box('test', 'test', 'string') == None

# def test_create_message_box():
#     '''Tests whether a message box can be created with valid arguments'''
#     assert gui_fx.create_message_box('title', 'test', wx.OK) == True

def test_draw_buttonshouldFail():
    '''Tests whether a button with invalid arguments is drawn'''
    assert gui_fx.draw_button('test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test') == False

def test_draw_button():
    '''Tests whether a button with valid arguments is drawn'''
    assert gui_fx.draw_button(10, 10, 10, 10, (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), 'test', 10, 0) == True

def test_draw_barShouldFail():
    '''Tests whether a scroll bar with invalid arguments is drawn'''
    assert gui_fx.draw_bar('test', 'test', 'test') == False

def test_draw_bar():
    '''Tests whether a scroll bar with valid arguments is drawn'''
    assert gui_fx.draw_bar(10, 10, gui_fx.ScrollBar(640, 0)) == True

def test_testValues():
    '''Tests whether testValues returns true for a function with valid arguments'''
    assert gui_fx.testValues({'test1': 10, 'test2': [10, 15, 24], 'test3': 'This is a string'}, ['test1', 'test2', 'test3'], [int, list, str], 'test_testValues') == True

def test_testValuesShouldFail():
    '''Tests whether testValues returns false for a function with invalid arguments'''
    assert gui_fx.testValues({'test1': 10, 'test2': [10, 15, 24], 'test3': 'This is a string'}, ['test1', 'test2', 'test3'], [int, str, str], 'test_testValues') == False

def test_addToContentHeight():
    '''Tests whether addToContentHeight sets a scroll bar's height to the correct value'''
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    gui_fx.scroll_bars[0][0].addToContentHeight(10)
    assert gui_fx.scroll_bars[0][0].contentHeight == 10

def test_resetContentHeight():
    '''Tests whether resetContentHeight resets a scroll bar's content height to its base content height'''
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    gui_fx.scroll_bars[0][0].addToContentHeight(10)
    gui_fx.scroll_bars[0][0].resetContentHeight()
    assert gui_fx.scroll_bars[0][0].contentHeight == 0

def test_translateBarToMovement():
    '''Tests whether a scroll bar correctly converts the amount it has been moved to the amount its content should move'''
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    gui_fx.scroll_bars[0][0].addToContentHeight(10)
    gui_fx.scroll_bars[0][0].translateBarToMovement(20)
    #only get first decimal place because the relative y pos here is a repeating decimal
    assert str(gui_fx.scroll_bars[0][0].relativeYPos)[:3] == "0.3"

def test_draw_switch():
    '''Tests whether a switch with correct parameters is drawn'''
    assert gui_fx.draw_switch(10, 10, gui_fx.BooleanSwitch(100, 100, 100, 100, 'test', 'false', 100, 'test'), 0) == True

def test_draw_switchShouldFail():
    '''Tests whether a switch with incorrect parameters can be drawn'''
    assert gui_fx.draw_switch(10, 10, (gui_fx.BooleanSwitch(100, 100, 100, 100, 'test', 'false', 100, 'test'), 0, 'test'), 0) == False

def test_draw_text():
    '''Tests whether text with correct parameters is drawn'''
    assert gui_fx.draw_text(10, 10, (0, 0, 0, 0), 10, 'test', 10) == True

def test_draw_textShouldFail():
    '''Tests whether text with incorrect parameters is drawn'''
    assert gui_fx.draw_text(10, 10, 'test', 10, 'test', 10) == False

def test_draw_input_fields():
    '''Tests whehter valid input fields are successfully drawn on the screen'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 100, 100, 100, 0, 16
    )
    assert main_window.draw_input_fields() == True
    main_window.close()

def test_draw_text_boxes():
    '''Tests whether valid text boxes are successfully drawn on the screen'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    gui_fx.add_text(
        100, 100, (247, 247, 247, 255), 100, "Parameters", 3, 3
    )
    assert main_window.draw_text_boxes() == True
    main_window.close()

def test_draw_scroll_bars():
    '''Tests whether valid scroll bars are successfully drawn on the screen'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    assert main_window.draw_scroll_bars() == True
    main_window.close()

def test_draw_switches():
    '''Tests whether valid switches are successfully drawn on the screen'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    gui_fx.add_switch(
        (gui_fx.BooleanSwitch(100, 100, 100, 100, 'test', True, 19, 'test'), 0, 0)
    )
    assert main_window.draw_switches() == True
    main_window.close()

def test_on_draw():
    '''Tests whether the window's on_draw function returns true'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    gui_fx.add_switch(
        (gui_fx.BooleanSwitch(100, 100, 100, 100, 'test', True, 19, 'test'), 0, 0)
    )
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    gui_fx.add_text(
        100, 100, (247, 247, 247, 255), 100, "Parameters", 3, 3
    )
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 100, 100, 100, 0, 16
    )
    gui_fx.addButton(
        0, 0, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    assert main_window.on_draw() == True
    main_window.close()

def test_draw_images():
    '''Tests whether valid images are successfully drawn on the screen'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.add_window(main_window)
    gui_fx.add_image(
        100, 100, "static/images/OHSU-RGB-4C-REV.png", 0,
        float(39), float(67), False
    )
    assert main_window.draw_images() == True
    main_window.close()

def test_insertSymbolAtIndex():
    '''Tests whether a symbol can be inserted into a string at a given index'''
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 100, 100, 100, 0, 16
    )
    gui_fx.inputFields[0][0].text = "here is some test text"
    gui_fx.typingCursorPos = 7
    gui_fx.insertSymbolAtIndex("n't", 0)
    assert gui_fx.inputFields[0][0].text == "here isn't some test text"

def test_searchParameters():
    '''Tests whether the searchParameters function finds a given input field'''
    gui_fx.add_input(
        gui_fx.InputField('test_id', False, False), 100, 610, 100, 100, 0, 16, 0
    )
    gui_fx.add_input(
        gui_fx.InputField('test_id2', False, False), 100, 580, 100, 100, 0, 16, 0
    )
    gui_fx.add_input(
        gui_fx.InputField('testsearch', False, False), 100, 580, 100, 100, 0, 16
    )
    gui_fx.inputFields[2][0].text = "testtwo"
    gui_fx.add_scroll((gui_fx.ScrollBar(640, 0), 0))
    gui_fx.scroll_bars[0][0].addToContentHeight(90)
    output = open(str('test.json'), 'w')
    output.write('{\n  "test_id": {\n    "readableName": "testone"\n  }, \n  "test_id2": {\n    "readableName": "testtwo"\n  }\n}')
    output.close()
    gui_fx.searchParameters("test.json", 0, "testsearch")
    os.remove('test.json')
    assert gui_fx.scroll_bars[0][0].yPos != 0
