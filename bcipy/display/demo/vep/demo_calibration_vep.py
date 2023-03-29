from psychopy import core, visual

from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.helpers.clock import Clock
from bcipy.display import InformationProperties, TaskDisplayProperties, VEPStimuliProperties

info = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Consolas'],
    info_text=['VEP Display Demo'],
)
task_display = TaskDisplayProperties(colors=['White'],
                                     font='Consolas',
                                     height=.1,
                                     text='1/4')

task_text = ['1/4', '2/4', '3/4', '4/4']
task_color = [['white'], ['white'], ['white'], ['white']]
num_boxes = 4
start_positions_for_boxes = [(-.3, -.3), (.3, -.3), (.3, .3), (-.3, .3)]

win = visual.Window(size=[700, 700], fullscr=False, screen=1, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=False,
                    winType='pyglet')
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print(f'Monitor refresh rate: {frameRate} Hz')

clock = core.Clock()
experiment_clock = Clock()
len_stimuli = 10
stimuli = VEPStimuliProperties(
    stim_color=[['white'] * num_boxes],
    stim_pos=start_positions_for_boxes,
    stim_height=0.1,
    stim_font='Consolas',
    timing=(1, 0.5, 4),  # prompt, fixation, stimuli
    stim_length=1,  # how many times to stimuli
)
vep = VEPDisplay(win, experiment_clock, stimuli, task_display, info)
timing = []
t = 2

# loop over the text and colors, present the stimuli and record the timing
for (txt, color) in zip(task_text, task_color):
    vep.update_task(txt, color[0])
    vep.schedule_to([['A', 'B'], ['Z'], ['P'], ['R', 'W']], [1, 0.5, 5], [['blue'], ['purple'], ['red'], ['white']])
    timing += vep.do_inquiry()

    # show the wait screen, this will only happen once
    while t > 0:
        t -= 1
        vep.wait_screen(f"Waiting for {t}s", color='white')
        core.wait(1)

print(timing)
win.close()
