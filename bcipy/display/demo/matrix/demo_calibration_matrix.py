"""Demo Matrix Display functionality related to Calibration task logic."""
# pylint: disable=invalid-name
from psychopy import core

from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay

info = InformationProperties(
    info_color=['white'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Matrix Calibration Demo'],
)

stim_properties = StimuliProperties(stim_font='Arial',
                                    stim_pos=(-0.6, 0.4),
                                    stim_height=0.17,
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

task_bar = CalibrationTaskBar(win, inquiry_count=4, current_index=0, font='Arial')
matrix_display = MatrixDisplay(win,
                               experiment_clock,
                               stim_properties,
                               task_bar=task_bar,
                               info=info)

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
