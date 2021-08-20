from psychopy import visual, core
from psychopy.iohub.devices import experiment
from psychopy.visual import text
from bcipy.display import InformationProperties, TaskDisplayProperties, StimuliProperties
from bcipy.display.matrix import MatrixDisplay
from bcipy.acquisition.marker_writer import NullMarkerWriter
from bcipy.helpers.task import alphabet

info = InformationProperties(
    info_color=['White'],
    info_pos=(-.5, -.75),
    info_height=[0.1],
    info_font=['Arial'],
    info_text='Calibration Demo',
)
task_display = TaskDisplayProperties(
    task_color=['White'],
    task_pos=(-.5, .8),
    task_font='Arial',
    task_height=.1,
    task_text='1/100'
)
stim_properties = StimuliProperties(
    stim_font='Arial',
    stim_pos=(-0.6, 0.4),
    stim_height=0.1,
    stim_inquiry=['A'],
    stim_colors=[],
    stim_timing=[0.1],
    is_txt_stim=True
)

# Initialize Stimulus
full_screen = False
static_clock = core.StaticPeriod()
experiment_clock = core.Clock()
win = visual.Window(size=[500, 500], fullscr=full_screen, screen=0, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=True,
                    winType='pyglet')
win.recordFrameIntervals = False

matrix_display = MatrixDisplay(
    win,
    static_clock,
    experiment_clock,
    stim_properties,
    task_display,
    info)


matrix_display.schedule_to(stimuli=['A', 'B', 'C'], timing=[0.5, 0.5, 0.5], colors=[])
matrix_display.update_task_state(text='1/100', color_list=['White'])
# matrix_display.draw_static()
win.flip()
core.wait(1)
matrix_display.do_inquiry()

matrix_display.schedule_to(stimuli=['X', 'F', '<'], timing=[1, 1, 1], colors=[])
matrix_display.update_task_state(text='2/100', color_list=['White'])
# matrix_display.draw_static()
win.flip()
core.wait(1)
matrix_display.do_inquiry()

# Flash a grid
matrix_display.build_grid()
matrix_display.window.flip()

core.wait(1)

# Flash a grid
matrix_display.build_grid()
matrix_display.window.flip()

# Remaining Items
# Add Task Text using the existing TaskDisplay and self.task (top of the screen 1/100 --> 2/100). Look at how RSVP does it.
# Add logic to update and draw static (Task text)
# Unit Tests
# Typing
