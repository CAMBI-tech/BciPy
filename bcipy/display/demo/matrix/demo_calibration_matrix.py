"""Demo Matrix Display functionality related to Calibration task logic."""
# pylint: disable=invalid-name
from psychopy import core
from bcipy.display import InformationProperties, TaskDisplayProperties, StimuliProperties
from bcipy.display.paradigm.matrix import MatrixDisplay

from bcipy.display import init_display_window

info = InformationProperties(
    info_color=['white'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Matrix Calibration Demo'],
)
task_display = TaskDisplayProperties(colors=['white'],
                                     font='Arial',
                                     height=.1,
                                     text='100')
stim_properties = StimuliProperties(stim_font='Arial',
                                    stim_pos=(-0.6, 0.4),
                                    stim_height=0.1,
                                    is_txt_stim=True)

# Initialize Stimulus
window_parameters = {
    'full_screen': False,
    'window_height': 500,
    'window_width': 500,
    'stim_screen': 0,
    'background_color': 'black'
}

experiment_clock = core.Clock()
win = init_display_window(window_parameters)
win.recordFrameIntervals = False

matrix_display = MatrixDisplay(win, experiment_clock, stim_properties,
                               task_display, info)

time_target = 2
time_fixation = 2
time_flash = 0.25
timing = [time_target] + [time_fixation] + [time_flash] * 5
colors = ['green', 'lightgray'] + ['white'] * 5
task_buffer = 2

matrix_display.schedule_to(stimuli=['A', '+', 'F', '<', 'A', 'B', 'C'],
                           timing=timing,
                           colors=colors)
matrix_display.update_task_bar()
matrix_display.do_inquiry()
core.wait(task_buffer)

matrix_display.schedule_to(stimuli=['B', '+', 'F', '<', 'A', 'B', 'C'],
                           timing=timing,
                           colors=colors)
matrix_display.update_task_bar()
matrix_display.do_inquiry()
core.wait(task_buffer)

matrix_display.schedule_to(stimuli=['C', '+', 'F', '<', 'A', 'B', 'C'],
                           timing=timing,
                           colors=colors)
matrix_display.update_task_bar()
matrix_display.do_inquiry()
core.wait(task_buffer)

matrix_display.schedule_to(stimuli=['<', '+', 'F', '<', 'A', 'B', 'C'],
                           timing=timing,
                           colors=colors)
matrix_display.update_task_bar()
matrix_display.do_inquiry()
core.wait(task_buffer)
