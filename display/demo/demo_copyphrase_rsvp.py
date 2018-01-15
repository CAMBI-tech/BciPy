#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
from psychopy import visual, core

from display.rsvp_disp_modes import CopyPhraseTask
from helpers.triggers import _write_triggers_from_sequence_copy_phrase

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

# Bar Graph
show_bg = True
tr_pos_bg = (0, .5)
bl_pos_bg = (-1, -.5)
size_domain_bg = 7
color_bg_txt = 'red'
font_bg_txt = 'Arial'
color_bar_bg = 'green'

# Initialize Stimulus
is_txt_sti = True


if is_txt_sti:
    ele_sti = [
        ['+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-', 'L'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'R', '<', 'A', 'Z'],
        ['+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'R']]
    color_sti = [['red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white', 'white']] * 4
else:
    ele_sti = [['static/images/RSVP_images/A.jpg',
                'static/images/RSVP_images/Red_Cross.png',
                'static/images/RSVP_images/C.jpg',
                'static/images/RSVP_images/D.jpg',
                'static/images/RSVP_images/E.jpg',
                'static/images/RSVP_images/P.jpg'],
               ['static/images/RSVP_images/T.jpg',
                'static/images/RSVP_images/Red_Cross.png',
                'static/images/RSVP_images/B.jpg',
                'static/images/RSVP_images/C.jpg',
                'static/images/RSVP_images/D.jpg',
                'static/images/RSVP_images/E.jpg']]

time_flash = .25
time_target = 2
time_cross = .6

timing_sti = [[time_cross] + [time_flash] * (len(ele_sti[0]) - 1)] * 4

# Dummy Bar Graph Params
dummy_bar_schedule_t = [['A', 'B', 'C', 'D', '<', '-', 'G'],
                        ['A', 'B', 'C', 'D', '<', 'H', 'G'],
                        ['A', 'B', 'C', 'R', 'M', 'K', 'G'],
                        ['A', 'B', 'C', 'R', '<', 'Z', 'G']]
dummy_bar_schedule_p = [[1, 1, 1, 2, 3, 2, 3], [1, 1, 1, 2, 7, 2, 1],
                        [1, 1, 1, 2, 3, 2, 3], [1, 1, 2, 12, 1, 2, 1]]

task_text = ['COPY_PHA', 'COPY_PH']
task_color = [['white'] * 5 + ['green'] * 2 + ['red'],
              ['white'] * 5 + ['green'] * 2]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]

# Initialize Window
win = visual.Window(size=[500, 500], screen=0, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=True)
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print frameRate

# Initialize Clock
clock = core.StaticPeriod(screenHz=frameRate)
experiment_clock = core.MonotonicClock(start_time=None)

rsvp = CopyPhraseTask(window=win, clock=clock, experiment_clock=experiment_clock, static_text_task=text_task,
                      static_color_task=color_task,
                      text_info=text_text,
                      color_info=color_text, pos_information=pos_text,
                      height_information=txt_height,
                      font_information=font_text,
                      color_task=['white'] * 4 + ['green'] * 2 + ['red'],
                      font_task=font_task, text_task='COPY_PH',
                      height_task=height_task,
                      font_sti=font_sti, pos_sti=pos_sti,
                      sti_height=sti_height,
                      stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
                      time_list_sti=[3] * 10,
                      tr_pos_bg=tr_pos_bg, bl_pos_bg=bl_pos_bg,
                      size_domain_bg=size_domain_bg,
                      color_bg_txt=color_bg_txt, font_bg_txt=font_bg_txt,
                      color_bar_bg=color_bar_bg,
                      is_txt_sti=is_txt_sti)

counter = 0

# uncomment trigger_file lines for demo with triggers!
# trigger_file = open('copy_phrase_triggers.txt','w')
for idx_o in range(len(task_text)):

    rsvp.bg.reset_weights()
    rsvp.update_task_state(text=task_text[idx_o], color_list=task_color[idx_o])
    rsvp.draw_static()
    win.flip()
    rsvp.sti.height = sti_height

    for idx in range(int(len(ele_sti) / 2)):
        # Schedule a sequence
        rsvp.stim_sequence = ele_sti[counter]
        if is_txt_sti:
            rsvp.color_list_sti = color_sti[counter]

        rsvp.time_list_sti = timing_sti[counter]

        #
        core.wait(.4)
        sequence_timing = rsvp.do_sequence()

        # _write_triggers_from_sequence_copy_phrase(sequence_timing,
        #                                           trigger_file, text_task,
        #                                           task_text[idx_o])

        # Get parameters from Bar Graph and schedule
        rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[counter],
                            weight=dummy_bar_schedule_p[counter])

        core.wait(.5)
        if show_bg:
            rsvp.show_bar_graph()

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
