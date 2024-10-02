"""Demo using the CopyPhraseDisplay. This logic is integrated as an RSVP task, but this demonstates
    how to use it in a custom way or can be used to help test new features before integrating.
"""

from psychopy import core

from bcipy.display import (
    init_display_window,
    InformationProperties,
    PreviewInquiryProperties,
    StimuliProperties)
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.paradigm.rsvp.mode.copy_phrase import CopyPhraseDisplay
from bcipy.helpers.clock import Clock

# Initialize Stimulus
is_txt_stim = True
show_preview_inquiry = True

# Inquiry preview
preview_inquiry_length = 5
preview_inquiry_key_input = 'space'
preview_inquiry_progress_method = 1  # press to accept ==1 wait to accept ==2
preview_inquiry_isi = 3

info = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Dummy Message'],
)

# Stimuli
time_flash = .25
time_prompt = 2
time_fixation = .6
len_stimuli = 10
inter_stim_buffer = .5
stimuli = StimuliProperties(
    stim_font='Arial',
    stim_pos=(0, 0),
    stim_height=0.6,
    stim_inquiry=['a'] * len_stimuli,
    stim_colors=['white'] * len_stimuli,
    stim_timing=[3] * len_stimuli,
    is_txt_stim=is_txt_stim)

window_parameters = {
    'full_screen': False,
    'window_height': 500,
    'window_width': 500,
    'stim_screen': 1,
    'background_color': 'black'
}

# Create the stimuli to be presented, in real-time these stimuli will likely be given by a model or randomized
if is_txt_stim:
    ele_sti = [
        ['+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'R', '<', 'A'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A']]
    color_sti = [['red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white']] * 4

timing_sti = [[time_fixation] + [time_flash] * (len(ele_sti[0]) - 1)] * 4


spelled_text = ['COPY_PHA', 'COPY_PH']
task_color = [['white'] * 5 + ['green'] * 2 + ['red'],
              ['white'] * 5 + ['green'] * 2]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]

# Initialize Window
win = init_display_window(window_parameters)
# This is useful during time critical portions of the code, turn off otherwise
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print(frameRate)

# Initialize Clock
clock = core.StaticPeriod(screenHz=frameRate)
experiment_clock = Clock()
task_bar = CopyPhraseTaskBar(win, task_text='COPY_PHRASE', font='Menlo')
preview_inquiry = PreviewInquiryProperties(
    preview_on=show_preview_inquiry,
    preview_only=True,
    preview_inquiry_length=preview_inquiry_length,
    preview_inquiry_key_input=preview_inquiry_key_input,
    preview_inquiry_progress_method=preview_inquiry_progress_method,
    preview_inquiry_isi=preview_inquiry_isi)
rsvp = CopyPhraseDisplay(
    win,
    clock,
    experiment_clock,
    stimuli,
    task_bar,
    info,
    preview_inquiry=preview_inquiry)

counter = 0

for idx_o in range(len(spelled_text)):

    rsvp.update_task_bar(text=spelled_text[idx_o])
    rsvp.draw_static()
    win.flip()

    for idx in range(int(len(ele_sti) / 2)):
        # Schedule a inquiry
        rsvp.stimuli_inquiry = ele_sti[counter]
        if is_txt_stim:
            rsvp.stimuli_colors = color_sti[counter]

        rsvp.stimuli_timing = timing_sti[counter]

        core.wait(inter_stim_buffer)

        if show_preview_inquiry:
            inquiry_timing, proceed = rsvp.preview_inquiry()
            print(inquiry_timing)
            if proceed:
                inquiry_timing.extend(rsvp.do_inquiry())
            else:
                print('Rejected inquiry! Handle here')
                inquiry_timing = rsvp.do_inquiry()
        else:
            inquiry_timing = rsvp.do_inquiry()

        core.wait(inter_stim_buffer)
        counter += 1

    # Get stimuli parameters
    rsvp.stim_inquiry = ele_list_dec[idx_o]
    rsvp.color_list_sti = ['green']
    rsvp.time_list_sti = [2]

win.close()
