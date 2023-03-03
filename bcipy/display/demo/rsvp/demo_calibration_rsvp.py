from psychopy import core

from bcipy.display.paradigm.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.display import InformationProperties, TaskDisplayProperties, StimuliProperties, init_display_window

info = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Calibration Demo'],
)
task_display = TaskDisplayProperties(colors=['White'],
                                     font='Arial',
                                     height=.1,
                                     text='100')

# Initialize Stimulus
is_txt_stim = True

if is_txt_stim:
    ele_sti = [
        ['B', '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-'],
        ['E', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A'],
        ['W', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A'],
        ['Q', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A']]
    color_sti = [['green', 'red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white']] * 4


time_flash = .25
time_prompt = 2
time_fixation = .6

timing_sti = [[time_prompt] + [time_fixation] + [time_flash] *
              (len(ele_sti[0]) - 1)] * 4

task_text = ['1/100', '2/100', '3/100', '4/100']
task_color = [['white'], ['white'], ['white'], ['white']]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]


window_parameters = {
    'full_screen': False,
    'window_height': 500,
    'window_width': 500,
    'stim_screen': 1,
    'background_color': 'black'
}
win = init_display_window(window_parameters)
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print(frameRate)

# Initialize Clock
clock = core.StaticPeriod(screenHz=frameRate)
experiment_clock = Clock()
len_stimuli = 10
stimuli = StimuliProperties(
    stim_font='Arial',
    stim_pos=(0, 0),
    stim_height=0.6,
    stim_inquiry=['a'] * len_stimuli,
    stim_colors=['white'] * len_stimuli,
    stim_timing=[3] * len_stimuli,
    is_txt_stim=is_txt_stim)
rsvp = CalibrationDisplay(
    win,
    clock,
    experiment_clock,
    stimuli,
    task_display,
    info)


for idx_o in range(len(task_text)):

    rsvp.update_task_bar(text=task_text[idx_o])
    rsvp.draw_static()
    win.flip()

    # Schedule a inquiry
    rsvp.stimuli_inquiry = ele_sti[idx_o]

    if is_txt_stim:
        rsvp.stimuli_colors = color_sti[idx_o]

    rsvp.stimuli_timing = timing_sti[idx_o]

    core.wait(.4)
    inquiry_timing = rsvp.do_inquiry()

    core.wait(.5)

# close the window and trigger_file
win.close()

# Print intervals
# intervalsMS = np.array(win.frameIntervals) * 1000
# print(intervalsMS)
