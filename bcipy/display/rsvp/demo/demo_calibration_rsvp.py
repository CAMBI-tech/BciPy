from psychopy import visual, core

from bcipy.display.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.triggers import _write_triggers_from_sequence_calibration
from bcipy.acquisition.marker_writer import NullMarkerWriter

# Initialize Stimulus Parameters
# Task Bar
color_task = 'White'
font_task = 'Times'
height_task = 0.1
text_task = '0/100'

# Text Bar
color_text = 'white'
font_text = 'Times'

pos_text = (0, -.75)
text_text = 'Demo For Calibration Task'
txt_height = 0.1

# Stimuli
font_sti = 'Times'
pos_sti = (0, 0)
sti_height = 0.6

# Initialize Stimulus
is_txt_stim = True

if is_txt_stim:
    ele_sti = [
        ['B', '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-', 'L'],
        ['E', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['W', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['Q', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z']]
    color_sti = [['green', 'red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white', 'white']] * 4


time_flash = .25
time_target = 2
time_cross = .6

timing_sti = [[time_target] + [time_cross] + [time_flash] *
              (len(ele_sti[0]) - 1)] * 4

task_text = ['1/100', '2/100', '3/100', '4/100']
task_color = [['white'], ['white'], ['white'], ['white']]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]

# Initialize Window
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

rsvp = CalibrationDisplay(
    win,
    clock,
    experiment_clock,
    marker_writer=NullMarkerWriter(),
    info_text=text_text,
    info_color=color_text,
    info_pos=pos_text,
    info_height=txt_height,
    info_font=font_text,
    task_color=['white'],
    task_font=font_task,
    task_text=task_text[0],
    task_height=height_task,
    stim_font=font_sti,
    stim_pos=pos_sti,
    stim_height=sti_height,
    stim_sequence=['a'] * 10,
    stim_colors=['white'] * 10,
    stim_timing=[3] * 10,
    is_txt_stim=is_txt_stim)

# uncomment trigger_file lines for demo with triggers!
# trigger_file = open('calibration_triggers.txt','w')
for idx_o in range(len(task_text)):

    rsvp.update_task_state(text=task_text[idx_o], color_list=task_color[idx_o])
    rsvp.draw_static()
    win.flip()
    rsvp.sti.height = sti_height

    # Schedule a sequence
    rsvp.stimuli_sequence = ele_sti[idx_o]

    if is_txt_stim:
        rsvp.stimuli_colors = color_sti[idx_o]

    rsvp.stimuli_timing = timing_sti[idx_o]

    core.wait(.4)
    sequence_timing = rsvp.do_sequence()

    # _write_triggers_from_sequence_calibration(sequence_timing, trigger_file)

    core.wait(.5)

# close the window and trigger_file
win.close()
# trigger_file.close()

# Print intervals
# intervalsMS = np.array(win.frameIntervals) * 1000
# print(intervalsMS)
