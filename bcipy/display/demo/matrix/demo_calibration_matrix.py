from psychopy import visual
from bcipy.display import InformationProperties, TaskDisplayProperties, StimuliProperties
from bcipy.acquisition.marker_writer import NullMarkerWriter

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
win.recordFrameIntervals = True