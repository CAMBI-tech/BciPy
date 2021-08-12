from psychopy import visual, core
from psychopy.visual import text
from bcipy.display import InformationProperties, TaskDisplayProperties, StimuliProperties
from bcipy.acquisition.marker_writer import NullMarkerWriter
from bcipy.helpers.task import alphabet

info = InformationProperties(
    info_color='White',
    info_pos=(-.5, -.75),
    info_height=0.1,
    info_font='Arial',
    info_text='Calibration Demo',
)
task_display = TaskDisplayProperties(
    task_color=['White'],
    task_pos=(-.5, .8),
    task_font='Arial',
    task_height=.1,
    task_text='1/100'
)

# Initialize Stimulus
is_txt_stim = True
full_screen = False
win = visual.Window(size=[500, 500], fullscr=full_screen, screen=0, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=True,
                    winType='pyglet')
win.recordFrameIntervals = False

# Properties
ALP = alphabet()
print(ALP)
x = -0.7
y = 0.5
inc = 0.2
xlimit = 0.7
flash_time = 0.25
pos = [x, y]
stim_registry = {}


def make_grid(pos):
    for sym in ALP:
        text_stim = visual.TextStim(win, text=sym, opacity=0.2, pos=(pos[0], pos[1]))
        stim_registry[sym] = text_stim
        text_stim.draw()
        pos[0] += inc
        if pos[0] >= xlimit:
            pos[1] -= inc
            pos[0] = x


for sym in ALP:
    make_grid([x, y])
    stim_registry[sym].opacity = 1
    stim_registry[sym].draw()
    win.flip()
    core.wait(flash_time)
    stim_registry[sym].opacity = 0.2
    stim_registry[sym].draw()
# core.wait(2)
# text_stim.opacity = 1
# text_stim.draw()
# win.flip()
# core.wait(0.5)
# text_stim.opacity = 0.2
# text_stim.draw()
# win.flip()
# core.wait(1)

# stim = visual.TextStim(win,
#                        text='A', pos=pos,
#                        wrapWidth=None, colorSpace='rgb',
#                        opacity=0.2, depth=-6.0)
# stim.draw()
# window.flip()
# # [1]. Use index and list: self.stimuli.append(stim) *** WHAT DATA STRUCTURE TO STORE THIS IS IN ORDER TO FLASH LATER!
# # [2]. Create a dictionary using the symbol as key and stimuli as value.
# self.stimuli['B'] = stimuli  # setting
