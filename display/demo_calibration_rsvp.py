#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division

from psychopy import visual, core

from rsvp_disp_modes import CalibrationTask
from trigger_helpers import _write_triggers_from_sequence_calibration

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

# Bar Graph
show_bg = 0
tr_pos_bg = (0, .5)
bl_pos_bg = (-1, -.5)
size_domain_bg = 7
color_bg_txt = 'red'
font_bg_txt = 'Arial'
color_bar_bg = 'green'

# Initialize Stimulus
# TODO: Find a smart way to discriminate image input from text
is_txt_sti = 1

if is_txt_sti:
    ele_sti = [
        ['B', '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-', 'L'],
        ['E', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['W', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['Q', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z']]
    color_sti = [['green', 'red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white', 'white']] * 4
else:
    ele_sti = [['.\RSVP_presentation_by_images\A.jpg',
                '.\RSVP_presentation_by_images\Red_Cross.png',
                '.\RSVP_presentation_by_images\C.jpg',
                '.\RSVP_presentation_by_images\D.jpg',
                '.\RSVP_presentation_by_images\E.jpg',
                '.\RSVP_presentation_by_images\P.jpg'],
               ['.\RSVP_presentation_by_images\T.jpg',
                '.\RSVP_presentation_by_images\Red_Cross.png',
                '.\RSVP_presentation_by_images\B.jpg',
                '.\RSVP_presentation_by_images\C.jpg',
                '.\RSVP_presentation_by_images\D.jpg',
                '.\RSVP_presentation_by_images\E.jpg']]

time_flash = .25
time_target = 2
time_cross = .6

timing_sti = [[time_target] + [time_cross] + [time_flash] *
              (len(ele_sti[0]) - 1)] * 4

# Dummy Bar Graph Params
dummy_bar_schedule_t = [['A', 'B', 'C', 'D', '<', '-', 'G'],
                        ['A', 'B', 'C', 'D', '<', 'H', 'G'],
                        ['A', 'B', 'C', 'R', 'M', 'K', 'G'],
                        ['A', 'B', 'C', 'R', '<', 'Z', 'G']]
dummy_bar_schedule_p = [[1, 1, 1, 2, 3, 2, 3], [1, 1, 1, 2, 7, 2, 1],
                        [1, 1, 1, 2, 3, 2, 3], [1, 1, 2, 12, 1, 2, 1]]

task_text = ['1/100', '2/100', '3/100', '4/100']
task_color = [['white'], ['white'], ['white'], ['white']]

# Initialize decision
ele_list_dec = [['[<]'], ['[R]']]

# Initialize Window
win = visual.Window(size=[500, 500], fullscr=False, screen=0, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg',
                    waitBlanking=True)
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print frameRate

# Initialize Clock
clock = core.StaticPeriod(screenHz=frameRate)
experiment_clock = core.MonotonicClock(start_time=None)

rsvp = CalibrationTask(window=win, clock=clock,
                       experiment_clock=experiment_clock,
                       text_information=text_text,
                       color_information=color_text, pos_information=pos_text,
                       height_information=txt_height,
                       font_information=font_text,
                       color_task=['white'],
                       font_task=font_task, text_task=task_text[0],
                       height_task=height_task,
                       font_sti=font_sti, pos_sti=pos_sti,
                       sti_height=sti_height,
                       ele_list_sti=['a'] * 10, color_list_sti=['white'] * 10,
                       time_list_sti=[3] * 10,
                       tr_pos_bg=tr_pos_bg, bl_pos_bg=bl_pos_bg,
                       size_domain_bg=size_domain_bg,
                       color_bg_txt=color_bg_txt, font_bg_txt=font_bg_txt,
                       color_bar_bg=color_bar_bg,
                       is_txt_sti=is_txt_sti)

file = open('calibration_trigger_file.txt','w') 
for idx_o in range(len(task_text)):

    rsvp.bg.reset_weights()
    rsvp.update_task_state(text=task_text[idx_o], color_list=task_color[idx_o])
    rsvp.draw_static()
    win.flip()
    rsvp.sti.height = sti_height

    # Schedule a sequence
    rsvp.ele_list_sti = ele_sti[idx_o]

    if is_txt_sti:
        rsvp.color_list_sti = color_sti[idx_o]
    rsvp.time_list_sti = timing_sti[idx_o]


    core.wait(.4)
    sequence_timing = rsvp.do_sequence()

    _write_triggers_from_sequence_calibration(sequence_timing, file)

    # Get parameters from Bar Graph and schedule
    rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[idx_o],
                        weight=dummy_bar_schedule_p[idx_o])

    core.wait(.5)

    if show_bg:
        rsvp.show_bar_graph()

# close the window and file
win.close()
file.close()

# Print intervals
# intervalsMS = np.array(win.frameIntervals) * 1000
# print(intervalsMS)
