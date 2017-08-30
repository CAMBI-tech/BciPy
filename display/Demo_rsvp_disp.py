#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
from psychopy import visual, core, event
from rsvp_disp import DisplayRSVP
import numpy as np


# Initialize Stimulus Parameters
# Task Bar
color_task = ['white','red','red','red','red']
font_task = 'Times'
pos_task = (-.8, .9)
text_task = '0/100'

# Text Bar
color_text = 'white'
font_text = 'Times'
pos_text = (-.5, -.75)
text_text = 'Dummy Message'
txt_height = 0.2

# Stimuli
pos_sti = (0, 0)
sti_height = 0.6

# Bar Graph
tr_pos_bg = (0, .5)
bl_pos_bg = (-1, -.5)
size_domain_bg = 7
color_bg_txt = 'red'
font_bg_txt = 'Arial'
color_bar_bg = 'green'
bg_max_num_step = 100

# Initialize Stimulus
# TODO: Find a smart way to discriminate image input from text
is_txt_sti = 1

if is_txt_sti:
    ele_sti = [
        ['C', '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-', 'L'],
        ['F', '+', 'F', 'G', 'E', 'C', 'S', 'Q', 'W', 'E', '<', '-', 'Z']]
    color_sti = [['green', 'red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white', 'white']] * 2


time_flash = .35
time_target = 2
time_cross = .6

timing_sti = [[time_target] + [time_cross] + [time_flash]
              * (len(ele_sti[0]) - 2)] * 2

# Dummy Bar Graph Params
dummy_bar_schedule_t = [['A', 'X', 'E', 'D', 'E', 'F', 'G'],
                        ['A', 'X', 'C', 'D', 'E', 'F', 'G']]
dummy_bar_schedule_p = [[0.49, 0.3, 0.1, 0.01, 0.05, 0.03, 0.02],
                        [0.03, 0.3, 0.01, 0.05, 0.49, 0.02, 0.1]]

# Initialize Window
win = visual.Window(size=[1440, 900], fullscr=True, screen=0, allowGUI=False,
                    allowStencil=False, monitor='testMonitor', color='black',
                    colorSpace='rgb', blendMode='avg', useFBO=True,
                    waitBlanking=True)
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()
visual.useFBO = False
print frameRate

# Initialize Clock
clock = core.StaticPeriod(screenHz=frameRate)

rsvp = DisplayRSVP(win, clock, color_task=color_task, font_task=font_task,
                   pos_task=pos_task, task_height=txt_height,
                   text_task=text_task, color_text=['white'],
                   text_text=['Information Text'], font_text=['Times'],
                   pos_text=[pos_text], height_text=[txt_height],
                   font_sti='Times', pos_sti=pos_sti, sti_height=sti_height,
                   tr_pos_bg=tr_pos_bg, bl_pos_bg=bl_pos_bg,
                   size_domain_bg=size_domain_bg, color_bg_txt=color_bg_txt,
                   font_bg_txt=font_bg_txt, color_bar_bg=color_bar_bg,
                   bg_step_num=bg_max_num_step, is_txt_sti=is_txt_sti)

for idx in range(len(ele_sti)):
    # Schedule a sequence
    rsvp.task.text = str(idx) + '/100'
    rsvp.ele_list_sti = ele_sti[idx]
    if is_txt_sti:
        rsvp.color_list_sti = color_sti[idx]
    rsvp.time_list_sti = timing_sti[idx]

    #
    core.wait(.5)

    rsvp.do_sequence()
    rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[idx],
                        weight=dummy_bar_schedule_p[idx])

    for idx_2 in range(rsvp.bg.max_num_step):
        rsvp.draw_static()
        rsvp.bg.animate(idx_2)
        win.flip()

    core.wait(.5)
win.close()

# Print intervals
intervalsMS = np.array(win.frameIntervals) * 1000
print(intervalsMS)
