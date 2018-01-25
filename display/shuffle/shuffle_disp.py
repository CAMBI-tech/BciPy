from psychopy import visual, core, event, logging
from utils.get_system_info import get_system_info


def flick(flicking_freqs, boxes, screen_refresh_rate, frameN):

    for index in range(len(boxes)):
        if frameN%round(screen_refresh_rate/flicking_freqs[index])==0:
            if boxes[index].fillColor == 'black':
                boxes[index].fillColor = 'white'
            else:
                boxes[index].fillColor = 'black'


'''Params:'''
flicking_freqs = [6, 10, 20, 30, 1, 1]
screen_refresh_rate = 60.
''''''

system_info = get_system_info()

# Can you use resolution to find smart positions for stimulus. 26/1920

win = visual.Window(system_info['RESOLUTION'],
                               monitor='testMonitor', units='deg')
frameN = 0
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


boxes = [box1, box2, box3, box4, box5, box6]

while True:
    box1.draw()
    box2.draw()
    box3.draw()
    box4.draw()
    box5.draw()
    box6.draw()
    win.update()
    frameN += 1

    flick(flicking_freqs=flicking_freqs, boxes=boxes,
          screen_refresh_rate=screen_refresh_rate, frameN=frameN)

    num_keys = len(event.getKeys())
    if num_keys > 0:
        print num_keys
        break
    event.clearEvents()

print('Overall, %i frames were dropped.' % win.nDroppedFrames)
print win.frameIntervals
print len(win.frameIntervals)

core.wait(.5)

win.close()
core.quit()