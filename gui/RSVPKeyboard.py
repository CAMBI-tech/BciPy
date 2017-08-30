from codecs import open as codecsopen
from json import load as jsonload
import pyglet
from warnings import warn
import gui_fx

main_window = gui_fx.MenuWindow(0, 'RSVP Keyboard')
gui_fx.addWindow(main_window)

#scroll bar ids are the same as the id of the window they attach to

mainWindowWidth = main_window.width
mainWindowHeight = main_window.height

#declare scroll bars.
#parameters: bar class(height of window), display window
#only add one scroll bar per window!
windowThreeBar = gui_fx.ScrollBar(mainWindowHeight)
windowZeroBar = gui_fx.ScrollBar(mainWindowHeight, visible=False)
windowFourBar = gui_fx.ScrollBar(mainWindowHeight)

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
    (247, 247, 247, 255), convertToWidth(22), "Parameters", 3, 3
)
gui_fx.addText(
    mainWindowWidthHalf + convertToWidth(10),
    mainWindowHeight - convertToHeight(20),
    (247, 247, 247, 255), convertToWidth(22), "Advanced Options", 4, 4
)
windowThreeBar.addToContentHeight(60)
windowFourBar.addToContentHeight(60)
with codecsopen("parameters.json", 'r', encoding='utf-8') as f:
    fileData = []
    try:
        fileData = jsonload(f)
    except ValueError:
        warn("Parameters file is formatted incorrectly!", Warning)
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
    #adds name of each parameter above its input box
    gui_fx.addText(
        mainWindowWidthHalf + convertToWidth(10),
        convertToHeight((sectionCounter) - (windowThreeBar.contentHeight \
        if (sectionBoolean) else windowFourBar.contentHeight)) + mainWindowHeight,
        (247, 247, 247, 255), convertToWidth(10), readableCaption,
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
        functionArg=("parameters.json", jsonItem),
        scrollBar=displayWindow, textSize=convertToWidth(14)
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
                convertToWidth(200), convertToHeight(40),
                jsonItem, valueBoolean, convertToWidth(19),
                 sectionString
            ), displayWindow, displayWindow)
        )
    else:
        #Adds an input field if an input field is needed
        gui_fx.addInput(
            gui_fx.InputField(jsonItem, sectionString),
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
                functionArg=[jsonItem, displayWindow, "parameters.json", "recommended_values"],
                scrollBar = displayWindow, textSize=convertToWidth(24)
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
f.close()

#because everything resizes dynamically, width/height/font size/position/etc
#values should be defined as a fraction of the main window's width or height.
#Some of the values below do this with a decimal value, while others divide a
#given size relative to the default window size by 480 (default window height)
#or 640 (default window width)

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

#Presentation mode button
gui_fx.addButton(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(50),
    convertToWidth(400), convertToHeight(75), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Presentation Mode', 2,
    functionCall="runPythonFile", functionArg=['testfile.py'],
    textSize=convertToWidth(22)
)
#View signals button
gui_fx.addButton(
    mainWindowWidthHalf, mainWindowHeightHalf - convertToHeight(50),
    convertToWidth(400), convertToHeight(75), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'View Signals', 2,
    functionCall="runPythonFile", functionArg=['testfile.py'],
    textSize=convertToWidth(22)
)
#Configure parameters button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(155),
    mainWindowHeightHalf - convertToHeight(150),
    convertToWidth(300), convertToHeight(70),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255),
    'Configure Parameters', 0, 3, textSize=convertToWidth(18)
)
#Save values button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(230),
    mainWindowHeightHalf - convertToHeight(100),
    convertToWidth(150), convertToHeight(60), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Save Values', 3,
    functionCall="writeValuesToFile", functionArg=(['bci_config', 'advanced_config'], valuesArray),
    textSize=convertToWidth(18)
)
#Load values button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(230),
    mainWindowHeightHalf - convertToHeight(170),
    convertToWidth(150), convertToHeight(60), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Load Values', 3,
    functionCall="readValuesFromFile", functionArg=(['bci_config', 'advanced_config'], valuesArray),
    textSize=convertToWidth(18)
)
#Advanced options button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(230),
    mainWindowHeightHalf + convertToHeight(140),
    convertToWidth(150), convertToHeight(50), (40, 40, 40, 255),
    (219, 219, 219, 255), (89, 89, 89, 255), 'Advanced Options', 3, 4,
    textSize=convertToWidth(12)
)
#Free spell button
gui_fx.addButton(
    mainWindowWidthHalf, mainWindowHeightHalf - convertToHeight(40),
     convertToWidth(100), convertToHeight(90),
     (25, 20, 1, 255), (239, 212, 105, 255), (255, 236, 160, 255), 'Free Spell',
      0, functionCall="setTrialType", functionArg=[3],
      textSize=convertToWidth(14)
)
#FRP Calibration button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(110),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (239, 146, 40, 255), (255, 190, 117, 255),
    'FRP Calibration', 0, functionCall="setTrialType", functionArg=[2],
    textSize=convertToWidth(14)
)
#Copy phrase button
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(110),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (117, 173, 48, 255), (186, 232, 129, 255), 'Copy Phrase',
    0, functionCall="setTrialType", functionArg=[4],
    textSize=convertToWidth(14)
)
#ERP calibration button
gui_fx.addButton(
    mainWindowWidthHalf - convertToWidth(220),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (221, 37, 56, 255), (245, 101, 71, 255),
    'ERP Calibration', 0, functionCall="setTrialType", functionArg=[1],
    textSize=convertToWidth(14)
)
#Mastery task button
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(220),
    mainWindowHeightHalf - convertToHeight(40),
    convertToWidth(100), convertToHeight(90),
    (25, 20, 1, 255), (62, 161, 232, 255), (81, 217, 255, 255), 'Mastery Task',
    0, functionCall="setTrialType", functionArg=[5],
    textSize=convertToWidth(14)
)
#Drop-down list button for user ids
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(122),
    mainWindowHeightHalf + convertToHeight(100),
    convertToWidth(40), convertToHeight(40),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), '', 0,
    functionCall="dropItems", functionArg=['user_id', 0, "users.txt", False],
    textSize=convertToWidth(24)
)
#Calculate AUC button
gui_fx.addButton(
    mainWindowWidthHalf + convertToWidth(155),
    mainWindowHeightHalf - convertToHeight(150),
    convertToWidth(300), convertToHeight(70),
    (40, 40, 40, 255), (219, 219, 219, 255), (89, 89, 89, 255), 'Calculate AUC',
     0, functionCall="runPythonFile", functionArg=['testfile.py'],
     textSize=convertToWidth(18)
)

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
    gui_fx.InputField('user_id', False), mainWindowWidthHalf,
    mainWindowHeightHalf + convertToHeight(100),
    convertToWidth(300), convertToHeight(50), 0,
    convertToWidth(16)
)

#register text to be displayed here.
#This is text displayed on the screen. Position and text size should be relative
#to window size.
#paramenters, x position, y position, color, size, text, display window, scroll
#bar id (if any)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(150),
    (247, 247, 247, 255), convertToWidth(19),
    "Enter existing or new user id:", 0
)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(200),
    (247, 247, 247, 255), convertToWidth(19),
    "RSVP Keyboard", 0
)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(50),
    (247, 247, 247, 255), convertToWidth(19), "Select type of trial:",
     0
)
gui_fx.addText(
    mainWindowWidthHalf, mainWindowHeightHalf + convertToHeight(150),
    (247, 247, 247, 255), convertToWidth(22), "Select Mode:", 2
)

#register images.
#Position, width, and height should be relative to the window size.
#parameters: x position, y position, file name, display window, width, height
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

if __name__ == '__main__':
    pyglet.app.run()
