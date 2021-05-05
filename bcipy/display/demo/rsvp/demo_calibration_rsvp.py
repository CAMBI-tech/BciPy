from psychopy import visual, core

from bcipy.display.rsvp.mode.calibration import CalibrationDisplay
from bcipy.display.rsvp import InformationProperties, TaskDisplayProperties, StimuliProperties
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

if is_txt_stim:
    ele_sti = [
        ['B', '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-'],
        ['E', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A'],
        ['W', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A'],
        ['Q', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A']]
    color_sti = [['green', 'red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white']] * 4


time_flash = .25
time_target = 2
time_cross = .6

timing_sti = [[time_target] + [time_cross] + [time_flash] *
              (len(ele_sti[0]) - 1)] * 4

task_text = ['1/100', '2/100', '3/100', '4/100']
task_color = [['white'], ['white'], ['white'], ['white']]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]

# Initialize Window TODO use initialize_display_window
win = visual.Window(size=[500, 500], fullscr=False, screen=0, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=True,
                    winType='pyglet')
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print(frameRate)

# Initialize Clock
clock = core.StaticPeriod(screenHz=frameRate)
experiment_clock = core.MonotonicClock(start_time=None)
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
    info,
    marker_writer=NullMarkerWriter())

# uncomment trigger_file lines for demo with triggers!
# trigger_file = open('calibration_triggers.txt','w')
for idx_o in range(len(task_text)):

    rsvp.update_task_state(text=task_text[idx_o], color_list=task_color[idx_o])
    rsvp.draw_static()
    win.flip()

    # Schedule a inquiry
    rsvp.stimuli_inquiry = ele_sti[idx_o]

    if is_txt_stim:
        rsvp.stimuli_colors = color_sti[idx_o]

    rsvp.stimuli_timing = timing_sti[idx_o]

    core.wait(.4)
    inquiry_timing = rsvp.do_inquiry()

    # _write_triggers_from_inquiry_calibration(inquiry_timing, trigger_file)

    core.wait(.5)

# close the window and trigger_file
win.close()
# trigger_file.close()

# Print intervals
# intervalsMS = np.array(win.frameIntervals) * 1000
# print(intervalsMS)
