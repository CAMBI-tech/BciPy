from __future__ import division
from os import listdir
from os.path import isfile
from psychopy import visual, core, event

# Path to pictures
#   ex: /Users/tabatha/Desktop/rsvp/
path = ""

# Get Image List
presentationImages = [f for f in listdir(path) if isfile(f)]


win = visual.Window(
    size=[1280, 800],
    fullscr=True,
    screen=0,
    allowGUI=False,
    allowStencil=False,
    monitor='testMonitor',
    color="#573534",
    colorSpace='rgb',
    useFBO=True,)

fixation = visual.GratingStim(
    win=win,
    mask='cross',
    size=0.2,
    pos=[0, 0],
    sf=0.1)

win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()
visual.useFBO = False
print frameRate

count_text = visual.TextStim(
    win,
    color='white',
    height=0.2,
    text="0/100",
    font='Times',
    pos=(-.8, .9),
    wrapWidth=None,
    colorSpace='rgb',
    opacity=1,
    depth=-6.0)

image = visual.ImageStim(
                    win,
                    mask=None,
                    units='',
                    pos=(0.0, 0.0),
                    size=None,
                    ori=0.0,
                    colorSpace='rgb',
                    contrast=1.0,
                    opacity=1.0,
                    depth=0,
                    interpolate=False,
                    flipHoriz=False,
                    flipVert=False,
                    texRes=128,
                    name=None,
                    autoLog=None,
                    maskParams=None
                    )
count_text.draw()
win.flip()

# Initialize clock and needed variables for looping
trialClock = core.Clock()
t = lastFPSupdate = 0
continueSequence = True
trial_index = 0
frameN = -1

# RSVP Presentation Parameters
presentationLength = .400
targetPresentationLength = 2
numberCalibrationTrials = 99 # count starts at 0

# Continues the loop until any key is pressed or continueSequence is False
while continueSequence and not event.getKeys():

    t = trialClock.getTime()
    frameN = frameN + 1
    test_index = 0

    for image_index in range(len(presentationImages)):
        # Make sure to ignore hidden file extensions and .py files within
        if not presentationImages[image_index].startswith('.') \
                                    and not presentationImages[image_index].endswith('.py'):

            # If this is the first letter, let's give user a chance to see it
            if test_index == 0:
                if trial_index > 0:
                    core.wait(targetPresentationLength)

                # Generate image from presentationImages
                image.image = presentationImages[image_index]
                image.tStart = t
                image.frameNStart = frameN

                # Update the count
                count_text.text = "%i/100" % (trial_index + 1)

                # Draw and refresh the screen
                image.draw()
                count_text.draw()
                test_index += 1
                win.flip()
                core.wait(targetPresentationLength)

            # If the second, show fixation
            elif test_index == 1:

                fixation.draw()
                count_text.draw()
                test_index += 1

                # refresh screen
                win.flip()
                core.wait(presentationLength)

            # Else, this greater than second image, just change image and
            #           present
            else:
                image.image = presentationImages[image_index]

                win.flip()
                image.draw()
                count_text.draw()
                test_index += 1

                core.wait(presentationLength)

    if trial_index == numberCalibrationTrials:
        continueSequence = False
    else:
        trial_index += 1

    win.flip()

win.close()
core.quit()
