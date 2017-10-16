
from os import environ
import pyglet
from warnings import warn
from sys import path
path.append('utility')
path.append('../io')
import utility.gui_fx as gui_fx
from load import load_json_parameters

main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
gui_fx.addWindow(main_window)

mainWindowWidth = main_window.width
mainWindowHeight = main_window.height

#declare scroll bars.
#parameters: bar class(height of window), id
windowThreeBar = gui_fx.ScrollBar(mainWindowHeight, 3)
windowZeroBar = gui_fx.ScrollBar(mainWindowHeight, 0, visible=False)
windowFourBar = gui_fx.ScrollBar(mainWindowHeight, 4)
optionsTabBar = gui_fx.ScrollBar(mainWindowWidth, 100, visible=False, horizontal=True)

mainWindowWidthHalf = int(mainWindowWidth/2)
mainWindowHeightHalf = int(mainWindowHeight/2)

def convertToHeight(inputNumber):
    return int(((inputNumber)/480.0)*mainWindowHeight)

def convertToWidth(inputNumber):
    return int(((inputNumber)/640.0)*mainWindowWidth)

#scrolling content
#Giving an item a scroll bar id causes the program to calculate its position
#based on the position of the scroll bar. The y position should represent the
#item's actual height before scrolling, and should still be relative to the
#height of the window.
#The scroll bar's content height is the height of the items scrolled by the bar.
# This is used to calculate how much they should scroll relative to the movement
#of the bar.
#
gui_fx.addText(
    mainWindowWidthHalf + convertToWidth(10),
    mainWindowHeight - convertToHeight(20),
    (247, 247, 247, 255), convertToWidth(20), "Parameters", 3, 3
)
gui_fx.addText(
    mainWindowWidthHalf + convertToWidth(10),
    mainWindowHeight - convertToHeight(20),
    (247, 247, 247, 255), convertToWidth(20), "Advanced Options", 4, 4
)
windowThreeBar.addToContentHeight(60)
windowFourBar.addToContentHeight(60)

path = "utility/parameters.json"
fileData = load_json_parameters(path)

counterbci = 0
counteradv = 0
#valuesArray contains the names of all the values in the config file, so that
#those names can be passed to the save/load data functions called by buttons.
valuesArray = []
for jsonItem in fileData:
    section = fileData[jsonItem]["section"]
    sectionBoolean = section == 'bci_config'
    displayWindow = 3 if fileData[jsonItem]["section"] == 'bci_config' else 4
    sectionCounter = counterbci if (sectionBoolean) else counteradv
    sectionString = 'bci_config' if (sectionBoolean) else 'advanced_config'
    if (sectionBoolean):
        counterbci = counterbci + 1
    else:
        counteradv = counteradv + 1
    readableCaption = fileData[jsonItem]["readableName"]
    isNumeric = fileData[jsonItem]["isNumeric"]
    #adds name of each parameter above its input box
    gui_fx.addText(
        mainWindowWidthHalf + convertToWidth(10),
        convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
        if (sectionBoolean) else windowFourBar.contentHeight)) + mainWindowHeight,
        (247, 247, 247, 255), convertToWidth(9), readableCaption,
        displayWindow, displayWindow
    )
    if (sectionBoolean):
        windowThreeBar.addToContentHeight(35)
    else:
        windowFourBar.addToContentHeight(35)
    #adds help button for each parameter
    gui_fx.addButton(
        mainWindowWidthHalf + convertToWidth(220),
        convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
            if (sectionBoolean) else windowFourBar.contentHeight)) + mainWindowHeight,
        convertToWidth(20), convertToHeight(20),
        (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '?',
        displayWindow, functionCall="displayHelpPointers",
        functionArg=("utility/parameters.json", jsonItem),
        scrollBar=displayWindow, textSize=convertToWidth(12)
    )
    value = fileData[jsonItem]["value"]
    if(value == 'true' or value == 'false'):
        valueBoolean = True if (value == 'true') else False
        #adds a switch instead of the input box for the parameter if it is a boolean
        gui_fx.addSwitch(
            (gui_fx.boolean_switch(
                mainWindowWidthHalf + convertToWidth(10),
                convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
                    if (sectionBoolean) else windowFourBar.contentHeight)) + mainWindowHeight,
                convertToWidth(200), convertToHeight(38),
                jsonItem, valueBoolean, convertToWidth(19),
                 sectionString
            ), displayWindow, displayWindow)
        )
    else:
        #Adds an input field if an input field is needed
        gui_fx.addInput(
            gui_fx.InputField(jsonItem, sectionString, True if isNumeric == "true" else False),
            mainWindowWidthHalf + convertToWidth(10),
            convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
            if (sectionBoolean) else windowFourBar.contentHeight)) + mainWindowHeight,
            convertToWidth(300), convertToHeight(40),
            displayWindow, convertToWidth(10), displayWindow
        )
        gui_fx.inputFields[len(gui_fx.inputFields) - 1][0].text = value
        #adds a drop-down list of recommended values for a parameter if it is needed
        if(fileData[jsonItem]["recommended_values"] != ''):
            gui_fx.addButton(
                mainWindowWidthHalf + convertToWidth(185),
                convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
                if (sectionBoolean) else windowFourBar.contentHeight)) + mainWindowHeight,
                convertToWidth(30), convertToHeight(30),
                (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '',
                displayWindow, functionCall="dropItems",
                functionArg=[jsonItem, displayWindow, "utility/parameters.json", "recommended_values"],
                scrollBar = displayWindow, textSize=convertToWidth(22)
            )
            gui_fx.addImage(
                mainWindowWidthHalf + convertToWidth(172),
                convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
                if (sectionBoolean) else windowFourBar.contentHeight) - 15) + mainWindowHeight,
                "static/images/triangle.png", displayWindow,
                float(convertToWidth(25)), float(convertToHeight(25)),
                displayWindow
            )
    valuesArray.append(jsonItem)
    if (sectionBoolean):
        windowThreeBar.addToContentHeight(35)
    else:
        windowFourBar.addToContentHeight(35)



#register all the buttons for all the windows here.
#registering a button with a given position, size, color scheme, caption, etc.
#with a given display window causes it to be shown and clickable when that
#window is opened.
#buttons can open another window if the window is declared as the window to open,
#and/or call a function with given arguments.
#windows do not need to be declared anywhere before being opened.
#parameters: x center pos (int), y center pos (int), width (int), height (int),
#text color (tuple), button color (tuple), outline color (tuple), caption (str or unicode),
#display window (int), window to open (int), function name to call (str),
#function arguments (list), text size (int), scroll bar id (int)
#Extend options menu
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(235),
    mainWindowHeightHalf + convertToHeight(210),
    convertToWidth(100), convertToHeight(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Show Options', 3,
    functionCall="moveMenu", functionArg=[100, convertToWidth(90)],
    textSize=convertToWidth(8)
)
#Presentation mode button
gui_fx.addButton(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(50),
    convertToWidth(400), convertToHeight(75), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Presentation Mode', 2,
    functionCall="runPythonFile", functionArg=['testing/testfile.py'],
    textSize=convertToWidth(20)
)
#View signals button
# gui_fx.addButton(
#     mainWindowWidthHalf, mainWindowHeightHalf - convertToHeight(50),
#     convertToWidth(400), convertToHeight(75), (40, 40, 40, 255),
#     (219, 219, 219, 255), (89, 89, 89, 255), 'View Signals', 2,
#     functionCall="runExecutable", functionArg=[environ['USERPROFILE'] + "\\Desktop", 'exe_name', True],
#     textSize=convertToWidth(20)
# )
#Configure parameters button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(155),
    mainWindowHeightHalf - convertToHeight(150),
    convertToWidth(300), convertToHeight(70),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255),
    'Configure Parameters', 0, 3, textSize=convertToWidth(16)
)
#Save values button
gui_fx.addButton(
    0,
    mainWindowHeightHalf - convertToHeight(100),
    convertToWidth(150), convertToHeight(60), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Save Values', 3,
    functionCall="writeValuesToFile", functionArg=(['bci_config', 'advanced_config'], valuesArray),
    textSize=convertToWidth(16), scrollBar=100
)
#Load values button
gui_fx.addButton(
    0,
    mainWindowHeightHalf - convertToHeight(170),
    convertToWidth(150), convertToHeight(60), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Load Values', 3,
    functionCall="readValuesFromFile", functionArg=(['bci_config', 'advanced_config'], valuesArray),
    textSize=convertToWidth(16), scrollBar=100
)
#Advanced options button
gui_fx.addButton(
    0,
    mainWindowHeightHalf + convertToHeight(30),
    convertToWidth(150), convertToHeight(50), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Advanced Options', 3, 4,
    textSize=convertToWidth(10), scrollBar=100
)
#Free spell button
gui_fx.addButton(
    mainWindowWidthHalf, mainWindowHeightHalf - convertToHeight(40),
     convertToWidth(100), convertToHeight(90),
     (25, 20, 1, 255), (239, 212, 105, 255), (255, 236, 160, 255), 'Free Spell',
      0, functionCall="setTrialType", functionArg=[3],
      textSize=convertToWidth(12)
)
#FRP Calibration button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(110),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (239, 146, 40, 255), (255, 190, 117, 255),
    'FRP Calibration', 0, functionCall="setTrialType", functionArg=[2],
    textSize=convertToWidth(12)
)
#Copy phrase button
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(110),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (117, 173, 48, 255), (186, 232, 129, 255), 'Copy Phrase',
    0, functionCall="setTrialType", functionArg=[4],
    textSize=convertToWidth(12)
)
#ERP calibration button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(220),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (221, 37, 56, 255), (245, 101, 71, 255),
    'ERP Calibration', 0, functionCall="setTrialType", functionArg=[1],
    textSize=convertToWidth(12)
)
#Mastery task button
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(220),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (62, 161, 232, 255), (81, 217, 255, 255), 'Mastery Task',
    0, functionCall="setTrialType", functionArg=[5],
    textSize=convertToWidth(12)
)
#Drop-down list button for user ids
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(122),
    mainWindowHeightHalf + convertToHeight(100),
    convertToWidth(40), convertToHeight(40),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0,
    functionCall="dropItems", functionArg=['user_id', 0, "users.txt", False],
    textSize=convertToWidth(22)
)
#Calculate AUC button
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(155),
    mainWindowHeightHalf - convertToHeight(150),
    convertToWidth(300), convertToHeight(70),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), 'Calculate AUC',
     0, functionCall="runPythonFile", functionArg=['testing/testfile.py'],
     textSize=convertToWidth(16)
)
#Search parameters button
gui_fx.addButton(
    0,
    mainWindowHeightHalf + convertToHeight(90),
    convertToWidth(60), convertToHeight(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Search', 3,
    functionCall="searchParameters", functionArg=['utility/parameters.json', 3, 'search'],
    textSize=convertToWidth(8), scrollBar=100
)
#Search advanced parameters button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(230),
    mainWindowHeightHalf + convertToHeight(90),
    convertToWidth(60), convertToHeight(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Search', 4,
    functionCall="searchParameters", functionArg=['utility/parameters.json', 4, 'advancedsearch'],
    textSize=convertToWidth(8)
)
#Retract options menu
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(325),
    mainWindowHeightHalf + convertToHeight(210),
    convertToWidth(100), convertToHeight(30), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Hide Options', 3,
    functionCall="moveMenu", functionArg=[100, convertToWidth(90)],
    textSize=convertToWidth(8), scrollBar=100
)
optionsTabBar.addToContentHeight(20)

#register all the input text fields for all the windows here.
#InputFields are passed a name as a parameter, which is used as a field name
#when the contents of the field are written to a text file.
#Like buttons, the width, height, and positions of input fields should be
#relative to the screen size. Input fields display in the window with their
#given display window id.
#parameters: inputField data, x center pos, y center pos, width, height, display
#window, text size, scroll bar id (if any)
#User id input field
gui_fx.addInput(
    gui_fx.InputField('user_id', False, False), mainWindowWidthHalf,
    mainWindowHeightHalf + convertToHeight(100),
    convertToWidth(300), convertToHeight(50), 0,
    convertToWidth(14)
)
#main parameters search menu
gui_fx.addInput(
    gui_fx.InputField('search', False, False), 0,
    mainWindowHeightHalf + convertToHeight(130), convertToWidth(150), convertToHeight(40), 3,
    convertToWidth(10), scrollBar=100
)
#advanced parameters search menu
gui_fx.addInput(
    gui_fx.InputField('advancedsearch', False, False), mainWindowWidthHalf - convertToWidth(230),
    mainWindowHeightHalf + convertToHeight(130), convertToWidth(150), convertToHeight(40), 4,
    convertToWidth(10)
)

#register text to be displayed here.
#This is text displayed on the screen. Position and text size should be relative
#to window size.
#paramenters, x position, y position, color, size, text, display window, scroll
#bar id (if any)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(150),
    (247, 247, 247, 255), convertToWidth(18),
    "Enter or select a user ID:", 0
)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(200),
    (247, 247, 247, 255), convertToWidth(18),
    "RSVP Keyboard", 0
)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(40),
    (247, 247, 247, 255), convertToWidth(18), "Select type of trial:",
     0
)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(150),
    (247, 247, 247, 255), convertToWidth(21), "Select Mode:", 2
)
gui_fx.addText(
    0, mainWindowHeightHalf + convertToHeight(170),
    (247, 247, 247, 255), convertToWidth(11), "Search Parameters", 3, scrollBar=100
)
gui_fx.addText(
    mainWindowWidthHalf - convertToWidth(230), mainWindowHeightHalf + convertToHeight(160),
    (247, 247, 247, 255), convertToWidth(8), "Search Advanced Parameters", 4
)

#register images.
#Position, width, and height should be relative to the window size.
#parameters: x position, y position, file name, display window, width, height, scroll bar
gui_fx.addImage(
    mainWindowWidthHalf + convertToWidth(260),
    mainWindowHeightHalf + convertToHeight(140),
    "static/images/OHSU-RGB-4C-REV.png", 0,
    float(convertToWidth(39)), float(convertToHeight(67)), False
)
gui_fx.addImage(
    mainWindowWidthHalf - convertToWidth(305),
    mainWindowHeightHalf + convertToHeight(115),
    "static/images/northeasternuniversity_logoseal.png", 0,
    float(convertToWidth(87)), float(convertToHeight(88)), False
)
gui_fx.addImage(
    mainWindowWidthHalf + convertToWidth(105),
    mainWindowHeightHalf + (80/480.0)*mainWindowHeight,
    "static/images/triangle.png", 0,
    float(convertToWidth(33)), float(convertToHeight(33)), False
)

#real scroll bar registration
#AddScroll takes the scroll bar itself and the id of the attached window as
#parameters.
gui_fx.addScroll((windowThreeBar, 3))
gui_fx.addScroll((windowZeroBar, 0))
gui_fx.addScroll((windowFourBar, 4))
gui_fx.addScroll((optionsTabBar, 3))

if __name__ == '__main__':
    pyglet.app.run()
