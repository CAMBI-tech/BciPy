from psychopy import visual, core

from bcipy.display.rsvp.mode.copy_phrase import CopyPhraseDisplay
from bcipy.helpers.triggers import _write_triggers_from_sequence_copy_phrase
from bcipy.acquisition.marker_writer import NullMarkerWriter

# Initialize Stimulus Parameters
# Task Bar
color_task = 'White'
font_task = 'Times'
height_task = 0.1
text_task = 'COPY_PHRASE'

# Text Bar
color_text = 'white'
font_text = 'Times'

pos_text = (-.5, -.75)
text_text = 'Dummy Message'
txt_height = 0.2

# Stimuli
font_sti = 'Times'
pos_sti = (0, 0)
sti_height = 0.6

# Initialize Stimulus
is_txt_stim = True


if is_txt_stim:
    ele_sti = [
        ['+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-', 'L'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'R', '<', 'A', 'Z'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'R']]
    color_sti = [['red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white', 'white']] * 4

time_flash = .25
time_target = 2
time_cross = .6

timing_sti = [[time_cross] + [time_flash] * (len(ele_sti[0]) - 1)] * 4


task_text = ['COPY_PHA', 'COPY_PH']
task_color = [['white'] * 5 + ['green'] * 2 + ['red'],
              ['white'] * 5 + ['green'] * 2]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]

# Initialize Window
win = visual.Window(size=[500, 500], screen=0, allowGUI=False,
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

rsvp = CopyPhraseDisplay(
    win,
    clock,
    experiment_clock,
    marker_writer=NullMarkerWriter(),
    static_task_text=text_task,
    static_task_color=color_task,
    info_text=text_text,
    info_color=color_text,
    info_pos=pos_text,
    info_height=txt_height,
    info_font=font_text,
    task_color=['white'],
    task_font=font_task, task_text='COPY_PH',
    task_height=height_task,
    stim_font=font_sti, stim_pos=pos_sti,
    stim_height=sti_height,
    stim_sequence=['a'] * 10, stim_colors=['white'] * 10,
    stim_timing=[3] * 10,
    is_txt_stim=is_txt_stim)

counter = 0

# uncomment trigger_file lines for demo with triggers!
# trigger_file = open('copy_phrase_triggers.txt','w')
for idx_o in range(len(task_text)):

    rsvp.update_task_state(text=task_text[idx_o], color_list=task_color[idx_o])
    rsvp.draw_static()
    win.flip()
    rsvp.sti.height = sti_height

    for idx in range(int(len(ele_sti) / 2)):
        # Schedule a sequence
        rsvp.stimuli_sequence = ele_sti[counter]
        if is_txt_stim:
            rsvp.stimuli_colors = color_sti[counter]

        rsvp.stimuli_timing = timing_sti[counter]

        #
        core.wait(.4)
        sequence_timing = rsvp.do_sequence()

        # _write_triggers_from_sequence_copy_phrase(sequence_timing,
        #                                           trigger_file, text_task,
        #                                           task_text[idx_o])

        core.wait(.5)
        counter += 1

    # Get stimuli parameters
    rsvp.stim_sequence = ele_list_dec[idx_o]
    rsvp.color_list_sti = ['green']
    rsvp.time_list_sti = [2]
    rsvp.do_sequence()

win.close()
# trigger_file.close()

# Print intervals
# intervalsMS = np.array(win.frameIntervals) * 1000
# print(intervalsMS)
