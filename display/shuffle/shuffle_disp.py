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

            self.contained_letters[index].pos = (self.position[0] + self.pos_shift*(index % 3) - 5,
                                                 self.position[1] - self.pos_shift*(index / 3) + 3)


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
flicking_freqs = [1, 2., 10., 20., 40, 60]
flick_duration = 2
screen_refresh_rate = 60.
colors = [(255, 255, 0), (255, 80, 80), (255, 0, 255), (51, 102, 255), (51, 204, 204), (51, 204, 51)]
alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']
'''##############################################################################'''

system_info = get_system_info()
win = visual.Window(system_info['RESOLUTION'],
                               monitor='testMonitor', units='deg')
win.recordFrameIntervals = True

# Create text boxes and rectangular boxes that cover the text boxes.
boxes = []
my_text_boxes = []
for index in range(6):
    boxes.append(visual.Rect(win = win, width=14, height=10, pos = (-21 + 21*(index%3), -10.8 + 21.6*(index/3)),
                   lineWidth=7, lineColorSpace='rgb255', lineColor=colors[index],
                   fillColor='black'))
    boxes[index].setAutoDraw(True)

    my_text_boxes.append(MyTextBox(contained_letters=None, position=(-21 + 21*(index%3), -10.8 + 21.6*(index/3)),
                                   pos_shift=5, color=colors[index]))

all_letters = []
for index in range(len(alp)):
    all_letters.append(TextStim(win=win, text=alp[index], pos=(-24.5 + 8*(index % 9), 4 - 3.8*(index / 9)), height=4, colorSpace='rgb255'))
    all_letters[index].setAutoDraw(True)

win.update()
core.wait(2)

while True:

    distribute_letters(all_letters, my_text_boxes)
    win.update()
    core.wait(2)

    [tb.set_pos() for tb in my_text_boxes]
    win.update()
    core.wait(2)

    flick_draw(win=win, flicking_freqs=flicking_freqs, boxes=boxes,
               screen_refresh_rate=screen_refresh_rate, flick_duration=flick_duration)

    num_keys = len(event.getKeys())
    if num_keys > 0:
        print num_keys
        break
    event.clearEvents()

print('Overall, %i frames were dropped.' % win.nDroppedFrames)
print '# of frames: ', len(win.frameIntervals)
print 'Frame intervals: ', win.frameIntervals
core.wait(.5)

win.close()
core.quit()





