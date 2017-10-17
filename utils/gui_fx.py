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

import pyglet
import wx

import gui_fx

#arrays of buttons, windows, text, input fields, scroll bars, etc on the screen
buttons = []
windows = []
inputFields = []
textBoxes = []
scrollBars = []
images = []
switches = []
horizontalScrolls = []
boxes = []
#cache of label objects
labelCache = {}

#index of buttons currently displayed on the screen
buttonsOnScreen = []

mainWindowWidth = 640
mainWindowHeight = 480

#which button is currently moused over, if any
mouseOnButton = 1000
#same for input fields
mouseOnInput = 1000
#which input field is currently active
currentActiveInputField = 1000
#which switch is moused over, if any
mouseOnSwitch = 1000
#shift key
shiftPressed = False
#which scroll bar is currently moused over, if any
scrollMouse = 1000
#position of the typing cursor relative to the length of the string
typingCursorPos = 0

#current help pointer string to be displayed by the mouse, if any
mouseHelp = False
#current mouse y pos
mouseY = 0
#current mouse x pos
mouseX = 0

#bci trial type
trialType = 0
#bci user id
userId = 0


#invisible wxpython window to show save file dialogs
wxApp = wx.App()
wxWindow = wx.Window()
wxWindow.Hide()

#scroll bar height
scrollBarHeight = 50

#Moves a scroll bar back and forth by a given amount. Used by the show/hide options buttons in the parameters window.
def moveMenu(barId, moveAmount):
    for eachBar in scrollBars:
        if(eachBar[0].scrollId == barId):
            if(eachBar[0].relativeYPos != moveAmount):
                moveScrollBar(barId, moveAmount)
            else:
                moveScrollBar(barId, eachBar[0].relativeYPos - moveAmount)

#Moves a scroll bar by a given amount, adjusting both the displayed bar and the bar's content's position accordingly.
def moveScrollBar(barId, moveAmount):
    for eachBar in scrollBars:
        if(eachBar[0].scrollId == barId):
            eachBar[0].relativeYPos = moveAmount
            barAmount = int((moveAmount*(eachBar[0].height - scrollBarHeight))/((eachBar[0].contentHeight/(640.0 if eachBar[0].isHorizontal else 480.0))*(mainWindowWidth if eachBar[0].isHorizontal else mainWindowHeight)))
            eachBar[0].yPos = barAmount

#searches human-readable parameter names for values that contain the text
#of a given input field
def searchParameters(readPath, scrollBar, inputName):
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == inputName):
            searchText = inputFields[counter][0].text
            counter = len(inputFields)
    if(searchText != ''):
        if(ospath.isfile(readPath)):
            with codecsopen(str(readPath), 'r', encoding='utf-8') as f:
                try:
                    fileData = jsonload(f)
                    searchItems = []
                    for jsonItem in fileData:
                        readableCaption = fileData[jsonItem]["readableName"]
                        if str.lower(str(searchText)) in str.lower(str(readableCaption)):
                            searchItems.append(jsonItem)
                    for eachBar in scrollBars:
                        if(eachBar[0].scrollId == scrollBar):
                            finalTempHeight = False
                            valueFound2 = False
                            selectedButton = False
                            selectedInput = False
                            prevYPos = eachBar[0].yPos
                            isButton = False
                            #repeats to find highest parameter if no lower parameter is found
                            for repeatCounter in range(0, 2):
                                valueFound = prevYPos if repeatCounter == 1 else False
                                if(repeatCounter == 0 or (selectedButton == False and selectedInput == False)):
                                    for counter in range(0, len(inputFields) + len(switches) - (1 if len(inputFields) != 0 else 0) - (1 if len(switches) != 0 else 0)):
                                        isInput = counter < len(inputFields)
                                        eachInput = inputFields[counter] if isInput else switches[counter - len(inputFields)]
                                        if((eachInput[7] if isInput else eachInput[2]) == scrollBar):
                                            for eachItem in searchItems:
                                                if((eachInput[0].name if isInput else eachInput[0].attachedValueName) == eachItem):
                                                    #Position that the scrolling content should be at in order to display the given parameter at the top of the window
                                                    tempHeight = abs((eachInput[2] if isInput else eachInput[0].centery) - int((385/480.0)*mainWindowHeight))
                                                    #Amount the scroll bar should move from the top in order to match the content height
                                                    barAmount = int((tempHeight*(eachBar[0].height - scrollBarHeight))/((eachBar[0].contentHeight/480.0)*mainWindowHeight))
                                                    if(repeatCounter == 0):
                                                        #tests if an input field has already been found (not looking for the lowest input field possible)
                                                        if(isInput and valueFound == False):
                                                            if(barAmount > prevYPos):
                                                                valueFound = barAmount
                                                                selectedInput = eachInput
                                                                finalTempHeight = tempHeight
                                                        elif(valueFound2 == False):
                                                            #tests if a switch has already been found, and if the new switch found is lower than the scroll bar's current position but higher than the previously found input field
                                                            if(barAmount > prevYPos and barAmount < valueFound):
                                                                valueFound2 = eachInput[2]
                                                                selectedButton = eachInput
                                                                finalTempHeight = tempHeight
                                                    else:
                                                        #looking for highest parameter in list
                                                        if(barAmount < valueFound):
                                                            valueFound = barAmount
                                                            finalTempHeight = tempHeight
                                                            if(isInput):
                                                                selectedInput = inputFields[counter]
                                                            else:
                                                                selectedButton = switches[counter - len(inputFields)]
                            if(finalTempHeight != False):
                                moveScrollBar(eachBar[0].scrollId, finalTempHeight)
                                #create yellow highlight rectangle
                                if(selectedButton != False):
                                    addButton(
                                        selectedButton[0].centerx, selectedButton[0].centery,
                                        selectedButton[0].width + int((10/640.0)*mainWindowWidth),
                                        int((50/480.0)*mainWindowHeight), (40, 40, 40, 255),
                                        (255, 215, 38, 5), (255, 204, 0, 255),
                                        '',
                                        scrollBar, scrollBar=scrollBar, isTemp=True
                                    )
                                elif(selectedInput != False):
                                    addButton(
                                        selectedInput[1], selectedInput[2],
                                        selectedInput[3] + int((10/640.0)*mainWindowWidth),
                                        int((50/480.0)*mainWindowHeight), (40, 40, 40, 255),
                                        (255, 215, 38, 5), (255, 204, 0, 255),
                                        '',
                                        scrollBar, scrollBar=scrollBar, isTemp=True
                                    )
                except ValueError:
                    warn('File ' + str(readPath) + " is an invalid JSON file.")
            f.close()
        else:
            warn("File " + str(readPath) + " could not be found.")


#for text boxes, inserts a given character (symbol) at the location indicated by
#typingcursorpos in the text of an input field located at inputFieldIndex in the global array
def insertSymbolAtIndex(symbol, inputFieldIndex):
    global typingCursorPos
    inputFields[inputFieldIndex][0].text = inputFields[inputFieldIndex][0].text[:typingCursorPos] + symbol + inputFields[inputFieldIndex][0].text[typingCursorPos:]
    typingCursorPos = typingCursorPos + 1

#changes the text of an input box. for drop-downs.
def changeInputText(inputName, changedText):
    global typingCursorPos
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == inputName):
            inputFields[counter][0].text = changedText
            counter = len(inputFields)
            typingCursorPos = len(changedText) - 1

#for drop-down menus. creates a drop-down of all items in a text file.
#textBoxName is the name of the parent input field. windowId is the name of the
#window that the buttons should be located in. fileName is the name of the file
#the values are read from. readValues should be true if the file is a JSON file.
def dropItems(textBoxName, windowId, fileName, readValues):
    global scrollBars
    for counter in range(0, len(scrollBars)):
        if(scrollBars[counter][0].scrollId == windowId):
            barIndex = counter
            counter = len(scrollBars)
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == textBoxName):
            if(ospath.isfile(fileName)):
                with codecsopen(fileName, 'r', encoding='utf-8') as f:
                    userArray = []
                    #determines whether content should be read as a json file
                    if(readValues == False):
                        userArray = f.readlines()
                    else:
                        try:
                            fileData = jsonload(f)
                            userArray = fileData[textBoxName][readValues]
                        except ValueError:
                            warn('File ' + str(fileName) + ' is an invalid JSON file.')
                    for counter2 in range(0, (len(userArray))):
                        if(isinstance(userArray[counter2], basestring)):
                            addButton(
                                inputFields[counter][1],
                                (inputFields[counter][2] - (inputFields[counter][4]) - ((counter2 - 1) * 10)) - counter2*20,
                                inputFields[counter][3],
                                int((20/480.0)*mainWindowHeight), (40, 40, 40, 255),
                                (219, 219, 219, 255), (89, 89, 89, 255),
                                userArray[counter2].replace("\n", '').replace("'u", ''),
                                windowId, functionCall="changeInputText",
                                functionArg=[textBoxName, userArray[counter2].replace("\n", '').replace("'u", '')],
                                scrollBar=windowId,
                                textSize=int((10/640.0)*mainWindowWidth), isTemp=True,
                                prioritizeTask=True, fontName='Arial'
                            )
                        else:
                            addButton(
                                inputFields[counter][1],
                                (inputFields[counter][2] - (inputFields[counter][4]) - ((counter2 - 1) * 10)) - counter2*20,
                                inputFields[counter][3],
                                int((20/480.0)*mainWindowHeight), (40, 40, 40, 255),
                                (219, 219, 219, 255), (89, 89, 89, 255),
                                str(userArray[counter2]).replace("\n", '').replace("'u", ''),
                                windowId, functionCall="changeInputText",
                                functionArg=[textBoxName, str(userArray[counter2]).replace("\n", '').replace("'u", '')],
                                scrollBar=windowId,
                                textSize=int((10/640.0)*mainWindowWidth), isTemp=True,
                                prioritizeTask=True, fontName='Arial'
                            )
                        scrollBars[barIndex][0].addToContentHeight((10/640.0)*mainWindowWidth)
                    f.close()
            else:
                warn("File " + str(fileName) + " could not be found.")

#sets trial type global, writes user id to file if the user id is not already in the file, and opens new window
def setTrialType(numId):
    global trialType
    global userId
    for counter in range(0, (len(inputFields))):
        if(inputFields[counter][0].name == 'user_id'):
            if(inputFields[counter][0].text != ''):
                trialType = numId
                userId = inputFields[counter][0].text
                new_window = MenuWindow(2, ' ')
                addWindow(new_window)
                if(ospath.isfile("users.txt")):
                    with codecsopen("users.txt", 'r', encoding='utf-8') as f:
                        userArray = f.readlines()
                        for eachItem in userArray:
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
                createMessageBox("Info", "Please enter a user ID.", wx.OK)

#reads a given file and looks for a help tip attached to a given id
def displayHelpPointers(fileName, helpId):
    global mouseHelp
    if(ospath.isfile(fileName)):
        with codecsopen(fileName, 'r', encoding='utf-8') as f:
            try:
                fileData = jsonload(f)
                mouseHelp = fileData[helpId]["helpTip"]
                f.close()
            except ValueError:
                warn('File ' + str(fileName) + ' was an invalid JSON file.')
    else:
        warn('File ' + str(fileName) + ' could not be found.')

#tests wheter a set of arguments passed to a function are of the correct types
def testValues(inputVariables, valueArray, typeArray, functionCaller):
    for counter in range(0, len(valueArray)):
        if(isinstance(typeArray[counter], list)):
            possibleVariableTypes = typeArray[counter]
            passed = False
            for variableType in possibleVariableTypes:
                if(isinstance(inputVariables.get(valueArray[counter]), variableType)):
                    passed = True
            if(passed == False):
                warn('Argument ' + str(valueArray[counter]) + ' passed to ' + str(functionCaller) + ' was of incorrect type' + str(type(inputVariables.get(valueArray[counter]))) + ' and should be of type ' + str(possibleVariableTypes))
                return False
        else:
            if(not isinstance(inputVariables.get(valueArray[counter]), typeArray[counter])):
                warn('Argument ' + str(valueArray[counter]) + ' passed to ' + str(functionCaller) + ' was of incorrect type ' + str(type(inputVariables.get(valueArray[counter]))) + ' and should be of type ' + str(typeArray[counter]))
                return False
    return True

#adds a button to the array of buttons
def addButton(xpos, ypos, width, height, tcolor, bcolor, lcolor, caption, displayWindow, openWindow=0, functionCall=0, functionArg=0, textSize=12, scrollBar=False, isTemp=False, prioritizeTask=False, fontName='Verdana'):
    global buttons
    shouldAdd = testValues(locals(), getargspec(addButton)[0], [int, int, int, int, tuple, tuple, tuple, [str, unicode], int, int, [int, str], [int, list, tuple], int, [bool, int], bool, bool, str], 'addButton')
    if(shouldAdd):
        return buttons.append((xpos, ypos, width, height, tcolor, bcolor, lcolor, caption, displayWindow, openWindow, functionCall, functionArg, textSize, scrollBar, isTemp, prioritizeTask, fontName))

def addWindow(newWindow):
    global windows
    shouldAdd = testValues(locals(), getargspec(addWindow)[0], [gui_fx.MenuWindow], 'addWindow')
    if(shouldAdd):
        return windows.append(newWindow)

def addInput(inputObject, xpos, ypos, width, height, windowid, textsize, scrollBar=False):
    global windows
    shouldAdd = testValues(locals(), getargspec(addInput)[0], [gui_fx.InputField, int, int, int, int, int, int, int], 'addInput')
    if(shouldAdd):
        return inputFields.append((inputObject, xpos, ypos, width, height, windowid, textsize, scrollBar))

def addText(xpos, ypos, color, size, text, window, scrollBar=False):
    global textBoxes
    shouldAdd = testValues(locals(), getargspec(addText)[0], [int, int, tuple, int, [str, unicode], int, [bool, int]], 'addText')
    if(shouldAdd):
        return textBoxes.append((xpos, ypos, color, size, text, window, scrollBar))

def addScroll(newBar):
    global scrollBars
    shouldAdd = testValues(locals(), getargspec(addScroll)[0], [tuple], 'addScroll')
    if(shouldAdd):
        newBar[0].baseContentHeight = newBar[0].contentHeight
        return scrollBars.append(newBar)

def addImage(centerx, centery, fileName, windowId, sizex, sizey, scrollBar):
    global images
    shouldAdd = testValues(locals(), getargspec(addImage)[0], [[int, float], [int, float], str, int, float, float, [int, bool]], 'addImage')
    if(shouldAdd):
        try:
            theImage = pyglet.image.load(fileName)
            sprite = pyglet.sprite.Sprite(theImage, x=centerx, y=centery)
            sprite.scale = sizex/theImage.width
            return images.append((sprite, windowId, scrollBar, centery, centerx))
        except IOError:
            warn("File '" + fileName + "' not found!")

def addSwitch(newSwitch):
    global switches
    shouldAdd = testValues(locals(), getargspec(addSwitch)[0], [tuple], 'addSwitch')
    if(shouldAdd):
        return switches.append(newSwitch)

#draws a button
def drawButton(centerx, centery, width, height, textColor, buttonColor, outlineColor, text, textSize, scrollId, fontName='Verdana'):
    if(testValues(locals(), getargspec(drawButton)[0], [[float, int], [float, int], [float, int], [float, int], tuple, tuple, tuple, [str, unicode], int, int, str], 'drawButton')):
        if(buttonColor[3] == 255):
            bottomColor = (buttonColor[0] - 15, buttonColor[1] - 15, buttonColor[2] - 15, buttonColor[3]) if ((buttonColor[0] - 15)  >= 0 and (buttonColor[1] - 15)  >= 0 and (buttonColor[2] - 15) >= 0) else buttonColor
            vertex_list = pyglet.graphics.vertex_list(4, ('v2f', ((centerx - (width/2)), (centery - (height/2)),
                (centerx + (width/2)), (centery - (height/2)),
                (centerx + (width/2)), (centery + (height/2)),
                (centerx - (width/2)), (centery + (height/2)))
                ),
                ('c4B', bottomColor + bottomColor + buttonColor + buttonColor)
                )
            vertex_list.draw(pyglet.gl.GL_QUADS)
        vertex_list = pyglet.graphics.vertex_list(4, ('v2f',
            ((centerx - (width/2)), (centery - (height/2)),
            (centerx + (width/2)), (centery - (height/2)),
            (centerx + (width/2)), (centery + (height/2)),
            (centerx - (width/2)), (centery + (height/2)))
            ),
            ('c4B', outlineColor + outlineColor + outlineColor + outlineColor)
            )
        vertex_list.draw(pyglet.gl.GL_LINE_LOOP)
        drawText(centerx, centery, textColor, textSize, text, scrollId, width, fontName)
        return True
    else:
        warn("Button with caption " + str(text) + " was not drawn.")
        return False

#parameters: width of window, height of window, scrollBar class
def drawBar(windowWidth, windowHeight, bar):
    if(testValues(locals(), getargspec(drawBar)[0], [int, int, gui_fx.ScrollBar], 'drawBar')):
        vertex_list = pyglet.graphics.vertex_list(4, ('v2f', ((windowWidth), (windowHeight),
        (windowWidth - (windowWidth/30)), (windowHeight),
        (windowWidth - (windowWidth/30)), (0),
        (windowWidth), (0))
        ),
        ('c4B', (173, 173, 173, 255, 173, 173, 173, 255, 173, 173, 173, 255, 173, 173, 173, 255))
        )
        vertex_list.draw(pyglet.gl.GL_QUADS)
        vertex_list = pyglet.graphics.vertex_list(4, ('v2f', ((windowWidth), (windowHeight - bar.yPos),
        (windowWidth - (windowWidth/30)), (windowHeight - bar.yPos),
        (windowWidth - (windowWidth/30)), (windowHeight - bar.yPos - scrollBarHeight),
        (windowWidth), (windowHeight - bar.yPos - scrollBarHeight))
        ),
        ('c4B', (235, 235, 235, 255, 235, 235, 235, 255, 235, 235, 235, 255, 235, 235, 235, 255))
        )
        vertex_list.draw(pyglet.gl.GL_QUADS)
        return True
    else:
        warn("Scroll bar failed to draw")
        return False

#draws a switch, with the button color varying depending on how the switch is set
def drawSwitch(windowWidth, windowHeight, switch, scrollId):
    if(testValues(locals(), getargspec(drawSwitch)[0], [int, int, gui_fx.boolean_switch, int], 'drawSwitch')):
        #temp y position that is altered if the switch is attached to a scroll bar
        centery = switch.centery
        centerx = switch.centerx
        if(scrollId != False):
            for eachScrollBar in scrollBars:
                if(eachScrollBar[0].scrollId == scrollId):
                    if(eachScrollBar[0].isHorizontal):
                        centerx = centerx + eachScrollBar[0].relativeYPos
                    else:
                        centery = centery + eachScrollBar[0].relativeYPos
        if(centery <= mainWindowHeight and centery >= 0):
            drawButton(
                switch.centerx, centery, switch.width, switch.height, \
                (30, 28, 24, 255), (48, 51, 50, 255), (15, 15, 14, 255), '', 1, \
                scrollId
            )
            drawButton(
                switch.centerx - int(switch.width/4), centery, \
                int(switch.width*0.45), int(switch.height*0.85), (30, 28, 24, 255) \
                 if switch.booleanValue else (242, 237, 208, 255), (102, 186, 50, 255) \
                 if switch.booleanValue else (27, 40, 11, 255), (15, 15, 14, 255), \
                 'True', switch.textSize, scrollId, 'Arial'
             )
            drawButton(
                switch.centerx + int(switch.width/4), centery, \
                int(switch.width*0.45), int(switch.height*0.85), \
                (242, 237, 208, 255)  if switch.booleanValue else (30, 28, 24, 255), \
                (56, 13, 7, 255) if switch.booleanValue else (219, 30, 30, 255), \
                (15, 15, 14, 255), 'False', switch.textSize, scrollId, 'Arial'
            )
            return True
    else:
        warn("Switch failed to draw")
        return False
#draws text
def drawText(centerx, centery, textColor, textSize, text, scrollId = False, buttonWidth = 0, fontName = 'Verdana'):
    global labelCache
    if(testValues(locals(), getargspec(drawText)[0], [[float, int], [float, int], tuple, int, [unicode, str], [int, bool], [int, float], str], 'drawText')):
        labelName = text + str(textSize) + str(textColor)
        if (labelName) in labelCache:
            if((isinstance(scrollId, bool))):
                labelCache[labelName].draw()
            else:
                labelCache[labelName].x=centerx
                labelCache[labelName].y=centery
                labelCache[labelName].anchor_x='center'
                labelCache[labelName].anchor_y='center'
                labelCache[labelName].draw()
        else:
            label = pyglet.text.Label(
                text, font_name = fontName, font_size=textSize, x=centerx, \
                y=centery, anchor_x='center', anchor_y='center', \
                color=textColor, multiline = True if buttonWidth != 0 else False, \
                width = buttonWidth if buttonWidth != 0 else None, align='center'
            )
            labelCache[labelName] = label
            label.draw()
        return True
    else:
        warn("Text with caption " + text + " failed to draw")
        return False

#writes the values of all current input boxes to a file. Parameters: name of config section, list of names of values to be written
def writeValuesToFile(sectionNames, fieldNames, fileName=None):
    try:
        dialog = wx.FileDialog(wxWindow, "Save Config As...", executable, ".json", "", wx.FD_SAVE)
        if(fileName == None):
            result = dialog.ShowModal()
            outputPath = dialog.GetDirectory() + "\\" + dialog.GetFilename()
        else:
            result = None
            outputPath = fileName
        if (result == wx.ID_OK or fileName != None):
            objects_list = OrderedDict()
            for counter3 in range(0, len(sectionNames)):
                sectionName = sectionNames[counter3]
                for counter in range(0, (len(fieldNames))):
                    d = OrderedDict()
                    for eachInputField in inputFields:
                        if(fieldNames[counter] == eachInputField[0].name and sectionName == eachInputField[0].sectionName):
                            d['value'] = eachInputField[0].text
                            objects_list[eachInputField[0].name] = d
                    for eachSwitch in switches:
                        if(eachSwitch[0].attachedValueName == fieldNames[counter] and sectionName == eachSwitch[0].sectionName):
                            d['value'] = 'true' if eachSwitch[0].booleanValue else 'false'
                            objects_list[eachSwitch[0].attachedValueName] = d
            if(ospath.isfile(outputPath)):
                output = open(str(outputPath), 'w')
            else:
                output = open(str(outputPath), 'a')
            j = jsondumps(objects_list, indent=2)
            output.write(j)
            output.close()
            dialog.Destroy()
    except TypeError:
        warn('Failed to create dialog')

#reads the values of all current input boxes from a file, then changes the input box text accordingly. Parameters: name of config section, list of names of values to be written.
def readValuesFromFile(sectionNames, fieldNames, fileName=None):
    global inputFields
    global switches
    try:
        dialog = wx.FileDialog(wxWindow, "Select Config File", executable, ".json", "", wx.FD_OPEN)
        if(fileName == None):
            result = dialog.ShowModal()
            readPath = dialog.GetDirectory() + "\\" + dialog.GetFilename()
        else:
            result = None
            readPath = fileName
        if (result == wx.ID_OK or fileName != None):
            if(ospath.isfile(readPath)):
                with codecsopen(str(readPath), 'r', encoding='utf-8') as f:
                    try:
                        fileData = jsonload(f)
                        for counter3 in range(0, len(sectionNames)):
                            sectionName = sectionNames[counter3]
                            for counter in range(0, (len(fieldNames))):
                                for counter2 in range(0, len(inputFields)):
                                    if(counter != len(fieldNames)):
                                        if(fieldNames[counter] == inputFields[counter2][0].name and sectionName == inputFields[counter2][0].sectionName):
                                            inputFields[counter2][0].text = fileData[inputFields[counter2][0].name]["value"]
                                            counter = len(fieldNames)
                                for counter2 in range(0, len(switches)):
                                    if(counter != len(fieldNames)):
                                        if(switches[counter2][0].attachedValueName == fieldNames[counter] and sectionName == switches[counter2][0].sectionName):
                                            switches[counter2][0].booleanValue = (True if (fileData[(fieldNames[counter])]["value"]) == 'true' else False)
                                            counter = len(fieldNames)
                    except ValueError:
                        warn('File ' + str(readPath) + " is an invalid JSON file.")
                    except KeyError:
                        warn('File ' + str(readPath) + " does not contain all parameter values.")
                f.close()
            else:
                warn("File " + str(readPath) + " could not be found.")
        dialog.Destroy()
    except TypeError:
        warn('Failed to create dialog')


#Creates a wxpython message box with a given title, text, and type.
#Currently does not do anything based on which button the user clicks.
def createMessageBox(title, text, thetype):
    try:
        dialog = wx.MessageDialog(wxWindow, text, title, thetype)
        result = dialog.ShowModal()
        if result == wx.ID_YES:
            pass
        dialog.Destroy()
        return True
    except TypeError:
        warn('Failed to create dialog with arguments: title: ' + title + " text: " + text + " type: " + str(thetype))

#runs a given python file
def runPythonFile(fileName):
    if(ospath.isfile(fileName)):
        try:
            execfile(fileName)
            return True
        except SyntaxError:
            warn("File " + str(fileName) + " is not a valid Python file.")
    else:
        warn("File " + str(fileName) + " not found.")

#Runs a command (fileName) from the given location (execPath). Intended to
#run exectuable files. If isParameter is true, fileName should be the name of the
#parameter containing the name of the exe file.
def runExecutable(execPath, fileName, isParameter):
    if(isParameter):
        for eachInputField in inputFields:
            if(fileName == eachInputField[0].name):
                fileName = eachInputField[0].text
    try:
        dir_path = ospath.dirname(ospath.realpath(__file__))
        chdir(execPath)
        subprocess.call(fileName)
        chdir(ospath.dirname(dir_path))
    except WindowsError:
        warn("Could not find executable " + fileName + " in path " + execPath)
        chdir(ospath.dirname(dir_path))

#removes drop-down items that are on the screen. Called when the user clicks
#anywhere after opening a drop-down list.
def removeDropDownList():
    buttonTrueIndex = []
    for counter in range(0, (len(buttons))):
        if(buttons[counter][14] == True):
            buttonTrueIndex.append(buttons[counter])
    for counter2 in range(0, (len(scrollBars))):
        for counter in range(0, len(buttonTrueIndex)):
            if(scrollBars[counter2][0].scrollId == buttonTrueIndex[counter][13]):
                if(scrollBars[counter2][0].baseContentHeight == 0):
                    scrollBars[counter2][0].yPos = 0
                scrollBars[counter2][0].resetContentHeight()
                scrollBars[counter2][0].translateBarToMovement(scrollBars[counter2][0].yPos)
                buttons.remove(buttonTrueIndex[counter])
    buttonTrueIndex = []

class boolean_switch():
    def __init__(self, par2xpos, par3ypos, par4width, par5height, par6name, par7defaultvalue, par8textsize, par9sectionName):
        self.centerx = par2xpos
        self.centery = par3ypos
        self.width = par4width
        self.height = par5height
        self.attachedValueName = par6name
        self.booleanValue = par7defaultvalue
        self.textSize = par8textsize
        self.sectionName = par9sectionName

#input box for the user to type in
class InputField():
    def __init__(self, par2Name, par3SectionName, par4IsNumeric):
        self.text = ""
        self.name = par2Name
        self.sectionName = par3SectionName
        self.isNumeric = par4IsNumeric

class ScrollBar():
    def __init__(self, par2WindowHeight, theID, relativeYPos = 0, visible=True, horizontal=False):
        #current position of the scroller
        self.yPos = 0
        #position of the content being scrolled
        self.relativeYPos = relativeYPos
        #actual height of the bar
        self.height = par2WindowHeight
        #height of the content being scrolled
        self.contentHeight = 0
        #is the scroll bar visible- used for scrolling drop-down lists
        self.isVisible = visible
        #the height of any content before drop-down menus are addedHeight
        self.baseContentHeight = 0
        #does the bar scroll horizontally?
        self.isHorizontal = horizontal
        #the id of this bar
        self.scrollId = theID

    def addToContentHeight(self, addedHeight):
        self.contentHeight = self.contentHeight + addedHeight

    def resetContentHeight(self):
        self.contentHeight = self.baseContentHeight

    #calculates how much the attached content needs to be moved based on the amount the bar has been scrolled
    def translateBarToMovement(self, barAmount):
        #a/b = c/d ad = bc
        #barAmount/height = x/contentHeight barAmount*contentHeight = height*x x=(barAmount*contentHeight)/height
        self.relativeYPos = (barAmount*(self.contentHeight/(640.0 if self.isHorizontal else 480.0))*(mainWindowWidth if self.isHorizontal else mainWindowHeight))/(self.height - scrollBarHeight)
        return self.relativeYPos

#window that opens when a button is pressed. window id is used to determine what should display in the window
class MenuWindow(pyglet.window.Window):
    def __init__(self, par2WindowId, par3Title):
        global mainWindowWidth
        global mainWindowHeight
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        self.windowId = par2WindowId
        mainWindowWidth = int((screen.width/3)*2)
        mainWindowHeight = int((screen.height/4)*3)
        super(MenuWindow, self).__init__(caption=(par3Title if(par3Title != ' ') else ("Window " + str(par2WindowId))), width=int((screen.width/3)*2), height=int((screen.height/4)*3))

    #draws all input fields
    def drawInputFields(self):
        global currentActiveInputField
        returnTrue = True
        for counter in range(0, (len(inputFields))):
            if(counter == currentActiveInputField):
                mainButtonColor = (207, 207, 207, 255)
                text = inputFields[counter][0].text[:typingCursorPos] + "|" + inputFields[counter][0].text[typingCursorPos:]
            else:
                mainButtonColor = (247, 247, 247, 255)
                text = inputFields[counter][0].text
            if(inputFields[counter][5] == self.windowId):
                scrollId = inputFields[counter][7]
                centery = inputFields[counter][2]
                centerx = inputFields[counter][1]
                if(scrollId != False):
                    for eachScrollBar in scrollBars:
                        if(eachScrollBar[0].scrollId == scrollId):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                tempWidth = inputFields[counter][3]
                if(tempWidth < len(text) * inputFields[counter][6] * 0.7):
                    tempWidth = len(text) * inputFields[counter][6] * 0.7
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    if(not drawButton(centerx, centery, tempWidth, inputFields[counter][4], (0, 0, 0, 255), mainButtonColor, (117, 117, 117, 255), text, inputFields[counter][6], scrollId, 'Arial')):
                        returnTrue = False
        return returnTrue

    #draws all text
    def drawTextBoxes(self):
        returnTrue = True
        for counter in range(0, (len(textBoxes))):
            if(textBoxes[counter][5] == self.windowId):
                scrollId = textBoxes[counter][6]
                centery = textBoxes[counter][1]
                centerx = textBoxes[counter][0]
                if(scrollId != False):
                    for eachScrollBar in scrollBars:
                        if(eachScrollBar[0].scrollId == scrollId):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    if(not drawText(centerx, centery, textBoxes[counter][2], textBoxes[counter][3], textBoxes[counter][4], textBoxes[counter][5])):
                        returnTrue = False
        return returnTrue

    #draws all buttons
    def drawButtons(self):
        global buttonsOnScreen
        returnTrue = True
        buttonsOnScreen = []
        for counter in range(0, (len(buttons))):
            if(mouseOnButton == counter):
                mainButtonColor = (max(0, buttons[counter][5][0] - 40), max(0, buttons[counter][5][1] - 40), max(0, buttons[counter][5][2] - 40), buttons[counter][5][3])
            else:
                mainButtonColor = buttons[counter][5]
            if(buttons[counter][8] == self.windowId):
                scrollId = buttons[counter][13]
                centery = buttons[counter][1]
                centerx = buttons[counter][0]
                if(not(isinstance(scrollId, bool))):
                    for eachScrollBar in scrollBars:
                        if(eachScrollBar[0].scrollId == scrollId):
                            if(eachScrollBar[0].isHorizontal):
                                centerx = centerx + eachScrollBar[0].relativeYPos
                            else:
                                centery = centery + eachScrollBar[0].relativeYPos
                if(centery < self.height and centery > 0 and centerx < self.width and centerx > 0):
                    buttonsOnScreen.append(counter)
                    if(not drawButton(centerx, centery, buttons[counter][2], buttons[counter][3], buttons[counter][4], mainButtonColor, buttons[counter][6], buttons[counter][7], buttons[counter][12], scrollId, buttons[counter][16])):
                        returnTrue = False
        return returnTrue

    #draws scroll bars
    def drawScrollBars(self):
        for counter in range(0, (len(scrollBars))):
            if(scrollBars[counter][0].scrollId == self.windowId and scrollBars[counter][0].isVisible == True):
                return drawBar(self.width, self.height, scrollBars[counter][0])

    #draws images
    def drawImages(self):
        global images
        returnTrue = True
        for counter in range(0, (len(images))):
            if(images[counter][1] == self.windowId):
                scrollId = images[counter][2]
                centery = images[counter][3]
                centerx = images[counter][4]
                if(not(isinstance(scrollId, bool))):
                    for eachScrollBar in scrollBars:
                        if(eachScrollBar[0].scrollId == scrollId):
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

    #draws switches
    def drawSwitches(self):
        returnTrue = True
        for counter in range(0, (len(switches))):
            if(switches[counter][1] == self.windowId):
                if(not drawSwitch(mainWindowWidth, mainWindowHeight, switches[counter][0], switches[counter][2])):
                    returnTrue = False
        return returnTrue

    #draws buttons, inputs, etc.
    def on_draw(self):
        global mouseX
        global mouseY
        returnTrue = True

        super(MenuWindow, self).clear()

        #background
        drawButton(mainWindowWidth/2, mainWindowHeight/2, mainWindowWidth, mainWindowHeight, (16, 19, 22, 255), (16, 19, 22, 255), (16, 19, 22, 255), '', 0, False)

        if(not self.drawInputFields()):
            returnTrue = False
        if(not self.drawTextBoxes()):
            returnTrue = False
        if(not self.drawSwitches()):
            returnTrue = False
        if(not self.drawButtons()):
            returnTrue = False
        if(not self.drawScrollBars()):
            returnTrue = False
        if(not self.drawImages()):
            returnTrue = False

        #draws a help tip box at the location of the mouse pointer if mouseHelp is set
        if(mouseHelp != False):
            if(not drawButton(int(mouseX - (50/640.0)*mainWindowWidth), int(mouseY - ((10*(len(mouseHelp)/10.0 + 1) + 15)/960.0)*mainWindowHeight), int((100/640.0)*mainWindowWidth), int(((10*(len(mouseHelp)/10.0 + 1) + 15)/480.0)*mainWindowHeight), (11, 4, 22, 255), (224, 217, 204, 255), (56, 55, 58, 255), mouseHelp, int((9/640.0)*mainWindowWidth), 0, fontName='Arial')):
                returnTrue = False
        return returnTrue


    #determines which, if any, button the mouse is over
    def on_mouse_motion(self, x, y, dx, dy):
        global mouseOnButton
        global mouseHelp
        global mouseX
        global mouseY
        changedMouse = False
        changedInput = False
        mouseX = x
        mouseY = y

        #Checks whethter the mouse is over a button, and, if so, changes the relevant variable
        for counter in range(0, (len(buttonsOnScreen))):
            buttonIndex = buttonsOnScreen[counter]
            if(buttonIndex < len(buttons)):
                if(self.windowId == buttons[buttonIndex][8]):
                    tempHeight = buttons[buttonIndex][1]
                    centerx = buttons[buttonIndex][0]
                    if(not(isinstance(buttons[buttonIndex][13], bool))):
                        for eachScrollBar in scrollBars:
                            if(eachScrollBar[0].scrollId == buttons[buttonIndex][13]):
                                if(eachScrollBar[0].isHorizontal):
                                    centerx = centerx + eachScrollBar[0].relativeYPos
                                else:
                                    tempHeight = tempHeight + eachScrollBar[0].relativeYPos
                    if((x <= (centerx + buttons[buttonIndex][2]/2)) and (x >= (centerx - buttons[buttonIndex][2]/2))):
                        if((y <= (tempHeight + buttons[buttonIndex][3]/2)) and (y >= (tempHeight - buttons[buttonIndex][3]/2))):
                            mouseOnButton = buttonIndex
                            changedMouse = True
                            counter = len(buttonsOnScreen)
        if(changedMouse == False):
            mouseOnButton = 1000
            mouseHelp = False

    #scroll bar handling
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        for counter in range(0, (len(scrollBars))):
            if(scrollBars[counter][0].scrollId == self.windowId):
                if(y < self.height and y > 0):
                    if((x <= self.width) and (x >= (self.width - (self.width / 30)))):
                        if((y <= self.height - scrollBars[counter][0].yPos + scrollBarHeight) and (y >= self.height - scrollBars[counter][0].yPos - scrollBarHeight)):
                            scrollBars[counter][0].yPos = self.height - y
                            if(scrollBars[counter][0].yPos < 0):
                                scrollBars[counter][0].yPos = 0
                            if(scrollBars[counter][0].yPos > self.height - scrollBarHeight):
                                scrollBars[counter][0].yPos = self.height - scrollBarHeight
                            scrollBars[counter][0].translateBarToMovement(self.height - y)
                return

    #more scroll bar handling
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        for counter in range(0, (len(scrollBars))):
            if(scrollBars[counter][0].scrollId == self.windowId):
                scrollBars[counter][0].yPos = scrollBars[counter][0].yPos - (scroll_y * 3)
                if(scrollBars[counter][0].yPos < 0):
                    scrollBars[counter][0].yPos = 0
                if(scrollBars[counter][0].yPos > self.height - scrollBarHeight):
                    scrollBars[counter][0].yPos = self.height - scrollBarHeight
                scrollBars[counter][0].translateBarToMovement(scrollBars[counter][0].yPos)
                return

    #opens the window attached to a button, or executes the function associated with the button
    def on_mouse_press(self, x, y, button, modifiers):
        global scrollBars
        global currentActiveInputField
        global switches
        global mouseOnSwitch
        global mouseOnInput
        global typingCursorPos

        #make sure that drop-down items can run their task before they are deleted
        if(mouseOnButton != 1000 and mouseOnButton < len(buttons)):
            if(buttons[mouseOnButton][15] == True):
                if(buttons[mouseOnButton][10] != 0):
                    getattr(gui_fx, buttons[mouseOnButton][10])(*(buttons[mouseOnButton][11]))

        removeDropDownList()

        if(currentActiveInputField != 1000):
            if(inputFields[currentActiveInputField][0].isNumeric == True):
                try:
                    float(inputFields[currentActiveInputField][0].text)
                except ValueError:
                    createMessageBox("Warning", "The parameter " + inputFields[currentActiveInputField][0].name + " takes a numeric value as input.", wx.ICON_EXCLAMATION)
            currentActiveInputField = 1000

        #run mouse tasks if necessary (window opening, function activation)
        if(mouseOnButton != 1000):
            if(mouseOnButton < len(buttons)):
                openWindow = True
                for counter in range(0, (len(windows))):
                    if(windows[counter].windowId == buttons[mouseOnButton][9]):
                        openWindow = False
                if(openWindow == True):
                    new_window = MenuWindow(buttons[mouseOnButton][9], ' ')
                    addWindow(new_window)
                if(buttons[mouseOnButton][10] != 0):
                    getattr(gui_fx, buttons[mouseOnButton][10])(*(buttons[mouseOnButton][11]))
        else:
            #detects whether a text box has been clicked
            changedInput = False
            for counter in range(0, (len(inputFields))):
                if(self.windowId == inputFields[counter][5]):
                    tempHeight = inputFields[counter][2]
                    centerx = inputFields[counter][1]
                    if(not(isinstance(inputFields[counter][7], bool))):
                        for eachScrollBar in scrollBars:
                            if(eachScrollBar[0].scrollId == inputFields[counter][7]):
                                if(eachScrollBar[0].isHorizontal):
                                    centerx = centerx + eachScrollBar[0].relativeYPos
                                else:
                                    tempHeight = tempHeight + eachScrollBar[0].relativeYPos
                    if((x <= (centerx + inputFields[counter][3]/2)) and (x >= (centerx - inputFields[counter][3]/2))):
                        if((y <= (tempHeight + inputFields[counter][4]/2)) and (y >= (tempHeight - inputFields[counter][4]/2))):
                            mouseOnInput = counter
                            changedInput = True
                            typingCursorPos = len(inputFields[counter][0].text)
                            counter = len(inputFields)
            if(changedInput == False):
                mouseOnInput = 1000
            if(mouseOnInput != 1000):
                currentActiveInputField = mouseOnInput
            else:
                #detects whether a switch has been clicked
                changedInput = False
                for counter in range(0, (len(switches))):
                    if(self.windowId == switches[counter][1]):
                        tempHeight = switches[counter][0].centery
                        centerx = switches[counter][0].centerx
                        if(not(isinstance(switches[counter][2], bool))):
                            for eachScrollBar in scrollBars:
                                if(eachScrollBar[0].scrollId == switches[counter][2]):
                                    if(eachScrollBar[0].isHorizontal):
                                        centerx = centerx + eachScrollBar[0].relativeYPos
                                    else:
                                        tempHeight = tempHeight + eachScrollBar[0].relativeYPos
                        if((x <= (centerx + switches[counter][0].width/2)) and (x >= (centerx - switches[counter][0].width/2))):
                            if((y <= (tempHeight + switches[counter][0].height/2)) and (y >= (tempHeight - switches[counter][0].height/2))):
                                mouseOnSwitch = counter
                                changedInput = True
                                counter = len(switches)
                if(changedInput == False):
                    mouseOnSwitch = 1000
                if(mouseOnSwitch != 1000):
                    switches[mouseOnSwitch][0].booleanValue = not(switches[mouseOnSwitch][0].booleanValue)


    #typing in input boxes
    def on_key_press(self, symbol, modifiers):
        global shiftPressed
        global currentActiveInputField
        global typingCursorPos

        #input field typing handling
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
                        insertSymbolAtIndex(pyglet.window.key.symbol_string(symbol), currentActiveInputField)
                    else:
                        insertSymbolAtIndex(str.lower(pyglet.window.key.symbol_string(symbol)), currentActiveInputField)
                elif(str.isdigit(pyglet.window.key.symbol_string(symbol)[1])):
                    insertSymbolAtIndex(pyglet.window.key.symbol_string(symbol)[1], currentActiveInputField)
                elif(symbol == pyglet.window.key.LSHIFT or symbol == pyglet.window.key.RSHIFT):
                    shiftPressed = True
                elif(symbol == pyglet.window.key.SPACE):
                    insertSymbolAtIndex(" ", currentActiveInputField)
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
                        insertSymbolAtIndex(pyglet.window.key.symbol_string(symbol)[4], currentActiveInputField)
                elif(symbol == pyglet.window.key.MINUS):
                    if(shiftPressed):
                        insertSymbolAtIndex("_", currentActiveInputField)
                    else:
                        insertSymbolAtIndex("-", currentActiveInputField)
                elif(symbol == pyglet.window.key.PERIOD):
                        insertSymbolAtIndex(".", currentActiveInputField)
                elif(symbol == pyglet.window.key.CAPSLOCK):
                    shiftPressed = (not shiftPressed)
                elif(symbol == pyglet.window.key.APOSTROPHE):
                        insertSymbolAtIndex("'", currentActiveInputField)
                elif(symbol == pyglet.window.key.BRACKETLEFT):
                    if(shiftPressed):
                        insertSymbolAtIndex("{", currentActiveInputField)
                    else:
                        insertSymbolAtIndex("[", currentActiveInputField)
                elif(symbol == pyglet.window.key.BRACKETRIGHT):
                    if(shiftPressed):
                        insertSymbolAtIndex("}", currentActiveInputField)
                    else:
                        insertSymbolAtIndex("]", currentActiveInputField)
            except IndexError:
                print "Invalid key press"


    #for detecting shift key usage for input box text
    def on_key_release(self, symbol, modifiers):
        global shiftPressed
        if(symbol == pyglet.window.key.LSHIFT or symbol == pyglet.window.key.RSHIFT):
            shiftPressed = False

    #removes window from window array so that it can be opened again, or, if this is the main window, closes all windows
    def on_close(self):
        for counter in range(0, (len(scrollBars))):
            if(scrollBars[counter][0].scrollId == self.windowId):
                scrollBars[counter][0].yPos = 0
                scrollBars[counter][0].translateBarToMovement(0)
        if(self.windowId == 0):
            for counter in range(0, len(windows)):
                if(windows[counter].windowId != self.windowId):
                    windows[counter].close()
            wxWindow.Close()
            super(MenuWindow, self).close()
            return
        #close advanced parameters window if main parameters window is closed
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
