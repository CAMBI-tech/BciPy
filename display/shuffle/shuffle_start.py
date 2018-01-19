from psychopy import visual, core, event
from utils.get_system_info import get_system_info

system_info = get_system_info()

# Can you use resolution to find smart positions for stimulus. 26/1920

shuffle_window = visual.Window(system_info['RESOLUTION'],
                               monitor='testMonitor', units='deg')

box1 = visual.Rect(win = shuffle_window, width=12, height=8, pos = (-21, -10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(255, 255, 0))

box2 = visual.Rect(win = shuffle_window, width=12, height=8, pos = (0, -10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(255, 80, 80))

box3 = visual.Rect(win = shuffle_window, width=12, height=8, pos = (21, -10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(255, 0, 255))

box4 = visual.Rect(win = shuffle_window, width=12, height=8, pos = (21, 10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(51, 102, 255))

box5 = visual.Rect(win = shuffle_window, width=12, height=8, pos = (0, 10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(51, 204, 204))

box6 = visual.Rect(win = shuffle_window, width=12, height=8, pos = (-21, 10.8),
                   lineWidth=4, lineColorSpace='rgb255', lineColor=(51, 204, 51))


while True:
    box1.draw()
    box2.draw()
    box3.draw()
    box4.draw()
    box5.draw()
    box6.draw()

    shuffle_window.update()

    num_keys = len(event.getKeys())
    if num_keys > 0:
        print num_keys
        break
    event.clearEvents()

core.wait(.5)

shuffle_window.close()
core.quit()