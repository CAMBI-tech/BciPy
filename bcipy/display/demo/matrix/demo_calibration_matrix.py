from psychopy import core
from bcipy.display import InformationProperties, TaskDisplayProperties, StimuliProperties
from bcipy.display.paradigm.matrix import MatrixDisplay

from bcipy.display import init_display_window

info = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Matrix Calibration Demo'],
)
task_display = TaskDisplayProperties(
    task_color=['White'],
    task_pos=(-.8, .85),
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
window_parameters = {
    'full_screen': False,
    'window_height': 500,
    'window_width': 500,
    'stim_screen': 1,
    'background_color': 'black'
}
static_clock = core.StaticPeriod()
experiment_clock = core.Clock()
win = init_display_window(window_parameters)
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
matrix_display.do_inquiry()

matrix_display.schedule_to(stimuli=['X', 'F', '<', 'A', 'B', 'C'], timing=[1, 1, 1, 1, 1, 1], colors=[])
matrix_display.update_task_state(text='2/100', color_list=['White'])
matrix_display.do_inquiry()

matrix_display.schedule_to(stimuli=['X', 'F', '<', 'A', 'B', 'C'], timing=[1, 1, 1, 1, 1, 1], colors=[])
matrix_display.update_task_state(text='3/100', color_list=['White'])
matrix_display.do_inquiry()

matrix_display.schedule_to(stimuli=['X', 'F', '<', 'A', 'B', 'C'], timing=[1, 1, 1, 1, 1, 1], colors=[])
matrix_display.update_task_state(text='4/100', color_list=['White'])
matrix_display.do_inquiry()
