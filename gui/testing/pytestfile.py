import codecs
import os
import pyglet
import pytest
import sys
sys.path.append('../')
import gui_fx

def teardown_function(function):
    gui_fx.inputFields = []
    gui_fx.scrollBars = []
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

def test_changeInputText():
    '''Tests that the function changeInputText can change the text of an input field'''
    gui_fx.addInput(
        gui_fx.InputField('test_id', False), 100, 100, 100, 100, 0, 16
    )
    gui_fx.changeInputText('test_id', "some_test_text")
    assert gui_fx.inputFields[0][0].text == "some_test_text"

def test_dropItems():
    '''Tests that the dropItems function can create a drop-down list of buttons based on values in a text or json file, and that those items can be removed'''
    gui_fx.addScroll((gui_fx.ScrollBar(640), 0))
    gui_fx.addInput(
        gui_fx.InputField('test_id', False), 100, 100, 100, 100, 0, 16
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
    gui_fx.removeDropDownList()
    #test whether removeDropDownList removes items
    assert len(gui_fx.buttons) == 0


def test_setTrialType():
    '''Tests that the setTrialType function can add a newly entered user to the users.txt file, and that it will not add that user more than once'''
    gui_fx.addInput(
        gui_fx.InputField('user_id', False), 100, 100, 100, 100, 0, 100
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
    gui_fx.addInput(
        gui_fx.InputField('testInputOne', 'test_section_1'), 100, 100, 100, 100, 0, 100
    )
    gui_fx.addInput(
        gui_fx.InputField('testInputTwo', 'test_section_2'), 100, 100, 100, 100, 0, 100
    )
    gui_fx.changeInputText('testInputOne', 'testone')
    gui_fx.changeInputText('testInputTwo', 'testtwo')
    gui_fx.writeValuesToFile(['test_section_1', 'test_section_2'], ['testInputOne', 'testInputTwo'], "test.txt")
    output = open(str('test.txt'), 'r')
    tempData = output.read()
    output.close()
    assert tempData == '{\n  "testInputOne": {\n    "value": "testone"\n  }, \n  "testInputTwo": {\n    "value": "testtwo"\n  }\n}'
    os.remove("test.txt")

def test_readValuesFromFile():
    '''Tests that the readValuesFromFile function readds the correct values from a json file and writes them to the correct input boxes'''
    gui_fx.addInput(
        gui_fx.InputField('testInputOne', 'test_section_1'), 100, 100, 100, 100, 0, 100
    )
    gui_fx.addInput(
        gui_fx.InputField('testInputTwo', 'test_section_2'), 100, 100, 100, 100, 0, 100
    )
    output = open(str('test.txt'), 'w')
    output.write('{\n  "testInputOne": {\n    "value": "testone"\n  }, \n  "testInputTwo": {\n    "value": "testtwo"\n  }\n}')
    output.close()
    gui_fx.readValuesFromFile(['test_section_1', 'test_section_2'], ['testInputOne', 'testInputTwo'], "test.txt")
    assert (gui_fx.inputFields[0][0].text == 'testone' and gui_fx.inputFields[1][0].text == 'testtwo')
    os.remove("test.txt")

def test_onMouseMotion():
    '''Tests that the on_mouse_motion function correctly determines whether the mouse cursor is over a button'''
    gui_fx.addButton(
        0, 0, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    gui_fx.mouseX = 50
    gui_fx.mouseY = 50
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.drawButtons()
    main_window.on_mouse_motion(50, 50, 0, 0)
    assert gui_fx.mouseOnButton == 0
    main_window.close()

def test_onMouseDrag():
    '''Tests that a scroll bar moves the appropriate amount after being dragged'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.addScroll((gui_fx.ScrollBar(main_window.height), 0))
    gui_fx.scrollBars[0][0].addToContentHeight(10)
    main_window.on_mouse_drag(main_window.width - 10, main_window.height - 10, 0, 0, 0, 0)
    assert gui_fx.scrollBars[0][0].relativeYPos == ((main_window.width - (main_window.width - 10))*(10/480.0)*main_window.height)/(main_window.height - gui_fx.scrollBarHeight)
    main_window.close()

def test_onMouseScroll():
    '''Tests that a scroll bar moves the appropriate amount relative to the mouse wheel'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.addScroll((gui_fx.ScrollBar(main_window.height), 0))
    gui_fx.scrollBars[0][0].addToContentHeight(10)
    main_window.on_mouse_scroll(100, 100, 0, -10)
    assert gui_fx.scrollBars[0][0].relativeYPos == (30*(10/480.0)*main_window.height)/(main_window.height - gui_fx.scrollBarHeight)
    main_window.close()

def test_onMousePress():
    '''Tests that the on_mouse_press function correctly determines whether the mouse cursor is over an input field or switch'''
    gui_fx.addInput(
        gui_fx.InputField('user_id', False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.addSwitch(
        (gui_fx.boolean_switch(100, 100, 100, 100, 'test', 'false', 100, 'test'), 0, 0)
    )
    gui_fx.mouseX = 50
    gui_fx.mouseY = 50
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.on_mouse_press(50, 50, 0, 0)
    #test input fields
    assert gui_fx.mouseOnInput == 0
    #test switches
    assert gui_fx.mouseOnSwitch == 0
    main_window.close()

def test_onKeyPress():
    '''Tests that a key press when an input field is active appends the correct letter to the field'''
    gui_fx.addInput(
        gui_fx.InputField('user_id', False), 100, 100, 100, 100, 0, 100
    )
    gui_fx.currentActiveInputField = 0
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.on_key_press(pyglet.window.key.A, None)
    assert gui_fx.inputFields[0][0].text == 'A'
    main_window.close()

def test_onKeyRelease():
    '''Tests shift key release'''
    gui_fx.addInput(
        gui_fx.InputField('user_id', False), 100, 100, 100, 100, 0, 100
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

def test_addInput():
    '''Tests whether an input field with correct parameters can be added'''
    gui_fx.addInput(
        gui_fx.InputField('test_id', False), 100, 100, 100, 100, 0, 16
    )
    assert len(gui_fx.inputFields) == 1

def test_addInputShouldFail():
    '''Tests wheter an input field with incorrect parameters can be aded. This should throw a warning.'''
    gui_fx.addInput(
        gui_fx.InputField('test_id', False), 'test', False, 100, 100, 0, 16
    )
    assert len(gui_fx.inputFields) == 0

def test_addText():
    '''Tests whether a text item with correct parameters can be added.'''
    gui_fx.addText(
        100, 100, (247, 247, 247, 255), 100, "Parameters", 3, 3
    )
    assert len(gui_fx.textBoxes) == 1

def test_addTextShouldFail():
    '''Tests whether a text item with incorrect parameters can be added. This should throw a warning.'''
    gui_fx.addText(
        100, 'test', (247, 247, 247, 255), 100, False, 3, 3
    )
    assert len(gui_fx.textBoxes) == 0

def test_addWindow():
    '''Tests whether a valid MenuWindow can be added'''
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    gui_fx.addWindow(main_window)
    assert len(gui_fx.windows) == 1
    main_window.close()

def test_addScroll():
    '''Tests whether a valid ScrollBar can be added'''
    gui_fx.addScroll((gui_fx.ScrollBar(640), 0))
    assert len(gui_fx.scrollBars) == 1

def test_addImage():
    '''Tests whether a valid image can be added'''
    gui_fx.addImage(
        100, 100, "../static/images/OHSU-RGB-4C-REV.png", 0,
        float(39), float(67), False
    )
    assert len(gui_fx.images) == 1

def test_addNonexistentImage():
    '''Tests whether an IOError is caught if an image is nonexistent. This should throw a warning.'''
    gui_fx.addImage(
        100, 100, "test.png", 0,
        float(39), float(67), False
    )
    assert len(gui_fx.images) == 0

def test_addSwitch():
    '''Tests whether a valid switch can be added'''
    gui_fx.addSwitch(
        (gui_fx.boolean_switch(100, 100, 100, 100, 'test', True, 19, 'test'), 0, 0)
    )

def test_drawButtons():
    '''Tests whether buttons on screen are drawn'''
    gui_fx.addButton(
        0, 0, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    gui_fx.addButton(
        0, 0, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    gui_fx.addButton(
        0, 0, 100, 100,
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0
    )
    main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
    main_window.drawButtons()
    assert len(gui_fx.buttonsOnScreen) == 3
    main_window.close()

def test_runPythonFileShouldFail():
    '''Tests whether an attempt is made to run an invalid python file from the runPythonFile function'''
    assert gui_fx.runPythonFile("../users.txt") == None
