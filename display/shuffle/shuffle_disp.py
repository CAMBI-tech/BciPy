from psychopy import visual, core, event
from utils.get_system_info import get_system_info
from psychopy.visual import TextStim
from operator import itemgetter

import numpy as np


class MyTextBox():
    contained_letters = None
    position = None
    pos_shift = None
    color = None

    def __init__(self, contained_letters, position, pos_shift, color):
        self.contained_letters = contained_letters
        self.position = position + (-3, 2)
        self.pos_shift = pos_shift
        self.color = color

    def set_letters(self, letters):
        self.contained_letters = letters

        for index in range(len(self.contained_letters)):
            self.contained_letters[index].color = self.color

    def set_pos(self):
        for index in range(len(self.contained_letters)):

            self.contained_letters[index].pos = (self.position[0] + self.pos_shift*(index % 3),
                                                 self.position[1] - self.pos_shift*(index / 3))


def flick_draw(win, flicking_freqs, boxes, screen_refresh_rate, flick_duration):

    frameN = 0

    while frameN < 1.*flick_duration*screen_refresh_rate:

        for index in range(len(boxes)):
            if frameN%round(screen_refresh_rate/flicking_freqs[index])==0:
                if boxes[index].fillColor == 'black':
                    boxes[index].fillColor = 'white'
                else:
                    boxes[index].fillColor = 'black'

            boxes[index].draw()

        win.update()

        frameN += 1


def distribute_letters(letters, myTextBoxes, posteriors = None):

    if posteriors:
        # Posteriors are provided, use a smart rule to distribute letters
        pass
    else:
        # Posteriors are not provided, demo mode, randomly shuffle

        split_indexes = np.split(np.array(np.random.permutation(range(len(letters)))), len(myTextBoxes))

        for index in range(len(myTextBoxes)):
            myTextBoxes[index].set_letters(itemgetter(*split_indexes[index])(letters))




'''Params:#######################################################################'''
# A second thought reveals that these are actually double the freq of flickering.
flicking_freqs = [6., 10., 20., 30., 1., 120]
flick_duration = .5
screen_refresh_rate = 60.

alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

'''##############################################################################'''

system_info = get_system_info()

win = visual.Window(system_info['RESOLUTION'],
                               monitor='testMonitor', units='deg')
win.recordFrameIntervals = True

box1 = visual.Rect(win = win, width=12, height=8, pos = (-21, -10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(255, 255, 0),
                   fillColor='black')
box2 = visual.Rect(win = win, width=12, height=8, pos = (0, -10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(255, 80, 80),
                   fillColor='black')
box3 = visual.Rect(win = win, width=12, height=8, pos = (21, -10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(255, 0, 255),
                   fillColor='black')
box4 = visual.Rect(win = win, width=12, height=8, pos = (21, 10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(51, 102, 255),
                   fillColor='black')
box5 = visual.Rect(win = win, width=12, height=8, pos = (0, 10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(51, 204, 204),
                   fillColor='black')
box6 = visual.Rect(win = win, width=12, height=8, pos = (-21, 10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(51, 204, 51),
                   fillColor='black')

text_box_1 = MyTextBox(contained_letters=None, position=(-21, -10.8),
                       pos_shift=3, color=(255, 255, 0))
text_box_2 = MyTextBox(contained_letters=None, position=(0, -10.8),
                       pos_shift=3, color=(255, 80, 80))
text_box_3 = MyTextBox(contained_letters=None, position=(21, -10.8),
                       pos_shift=3, color=(255, 0, 255))
text_box_4 = MyTextBox(contained_letters=None, position=(21, 10.8),
                       pos_shift=3, color=(51, 102, 255))
text_box_5 = MyTextBox(contained_letters=None, position=(0, 10.8),
                       pos_shift=3, color=(51, 204, 204))
text_box_6 = MyTextBox(contained_letters=None, position=(-21, 10.8),
                       pos_shift=3, color=(51, 204, 51))

my_text_boxes = [text_box_1, text_box_2, text_box_3, text_box_4, text_box_5, text_box_6]


boxes = [box1, box2, box3, box4, box5, box6]
for box in boxes:
    box.draw()

all_letters = []
for index in range(len(alp)):
    all_letters.append(TextStim(win=win, text=alp[index], pos=(-15 + 5*(index % 7), 5 - 3*(index / 7)), height=3, colorSpace='rgb255'))
    all_letters[index].setAutoDraw(True)

while True:

    flick_draw(win =win, flicking_freqs=flicking_freqs, boxes=boxes,
          screen_refresh_rate=screen_refresh_rate, flick_duration=flick_duration)

    distribute_letters(all_letters,my_text_boxes)

    for t_b in my_text_boxes:
        t_b.set_pos()

    num_keys = len(event.getKeys())
    if num_keys > 0:
        print num_keys
        break
    event.clearEvents()

print('Overall, %i frames were dropped.' % win.nDroppedFrames)
print '# of frames: ', len(win.frameIntervals)
print 'Frame intervals: ', win.frameIntervals
print 'Frame clock: ', win.frameClock.getTime()
core.wait(.5)

win.close()
core.quit()