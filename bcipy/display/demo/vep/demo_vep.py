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
task_display = TaskDisplayProperties(
    task_color=['White'],
    task_pos=(-.8, .85),
    task_font='Consolas',
    task_height=.1,
    task_text='1/4'
)

task_text = ['1/4', '2/4', '3/4', '4/4']
task_color = [['white'], ['white'], ['white'], ['white']]
num_boxes = 4
start_positions_for_boxes = [(-.3, -.3), (.3, -.3), (.3, .3), (-.3, .3)]

win = visual.Window(size=[700, 500], fullscr=False, screen=1, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=False,
                    winType='pyglet')
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print(frameRate)

# Initialize Clock
clock = core.Clock()
experiment_clock = Clock()
len_stimuli = 10
stimuli = VEPStimuliProperties(
    stim_color= [['white'] * num_boxes],
    stim_pos=start_positions_for_boxes,
    stim_height=0.1,
    stim_font='Consolas',
    timing=(1, 0.5, 4), # prompt, fixation, stimuli
    stim_length=1, # how many times to stimuli
)
vep = VEPDisplay(win, experiment_clock, stimuli, task_display, info)

for i in range(len(task_text)):
    vep.update_task(task_text[i], task_color[i][0])

    vep.do_inquiry()

    vep.schedule_to([['A', 'B'], ['Z'], ['P'], []], [1, 0.5, 5], [['white'], ['white'], ['white'], ['white']])
    # import pdb; pdb.set_trace()
    vep._generate_inquiry()
    import pdb; pdb.set_trace()

win.close()
