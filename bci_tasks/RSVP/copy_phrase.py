# Calibration Task for RSVP

from __future__ import division
from psychopy import core

import pdb

from display.rsvp_disp_modes import CopyPhraseTask
from helpers.trigger_helpers import _write_triggers_from_sequence_copy_phrase


def RSVP_copy_phrase_task(win, daq, parameters, file_save):
    # daq.start_acquistion(file_save)
    # Initialize Experiment clocks etc.
    frame_rate = win.getActualFrameRate()
    clock = core.StaticPeriod(screenHz=frame_rate)
    experiment_clock = core.MonotonicClock(start_time=None)

    try:
        rsvp = CopyPhraseTask(
            window=win, clock=clock,
            experiment_clock=experiment_clock,
            text_information=parameters['text_text']['value'],
            color_information=parameters['color_text']['value'],
            pos_information=(float(parameters['pos_text_x']['value']),
                             float(parameters['pos_text_y']['value'])),
            height_information=float(parameters['txt_height']['value']),
            font_information=parameters['font_text']['value'],
            color_task=['white'],
            font_task=parameters['font_task']['value'],
            height_task=float(parameters['height_task']['value']),
            font_sti=parameters['font_sti']['value'],
            pos_sti=(float(parameters['pos_sti_x']['value']),
                     float(parameters['pos_sti_y']['value'])),
            sti_height=float(parameters['sti_height']['value']),
            ele_list_sti=['a'] * 10, color_list_sti=['white'] * 10,
            time_list_sti=[3] * 10,
            # tr_pos_bg=parameters['tr_pos_bg']['value'],
            # bl_pos_bg=parameters['bl_pos_bg']['value'],
            size_domain_bg=int(parameters['size_domain_bg']['value']),
            color_bg_txt=parameters['color_bg_txt']['value'],
            font_bg_txt=parameters['font_bg_txt']['value'],
            color_bar_bg=parameters['color_bar_bg']['value'],
            is_txt_sti=parameters['is_txt_sti']['value'])
    except Exception as e:
        pdb.set_trace()
    # Init Task
    # run = True
    # while run is True:
        # to-do allow pausing and exiting. See psychopy getKeys()
    (task_text, task_color, ele_sti, timing_sti, color_sti) = get_task_info()
    try:
        counter = 0
    # file = open('copy_phrase_trigger_file.txt','w') 
        for idx_o in range(len(task_text)):

            rsvp.bg.reset_weights()
            rsvp.update_task_state(text=task_text[idx_o], color_list=task_color[idx_o])
            rsvp.draw_static()
            win.flip()
            rsvp.sti.height = float(parameters['sti_height']['value'])

            for idx in range(int(len(ele_sti) / 2)):
                # Schedule a sequence
                rsvp.ele_list_sti = ele_sti[counter]
                if parameters['is_txt_sti']['value']:
                    rsvp.color_list_sti = color_sti[counter]

                rsvp.time_list_sti = timing_sti[counter]

                #
                core.wait(.4)
                sequence_timing = rsvp.do_sequence()

                # _write_triggers_from_sequence_copy_phrase(sequence_timing, file, text_task, task_text[idx_o])

                # # Get parameters from Bar Graph and schedule
                # rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[counter],
                #                     weight=dummy_bar_schedule_p[counter])

                core.wait(.5)
                # if show_bg:
                #     rsvp.show_bar_graph()

                counter += 1

            # Get stimuli parameters
            # rsvp.ele_list_sti = ele_list_dec[idx_o]
            rsvp.color_list_sti = ['green']
            rsvp.time_list_sti = [2]
            rsvp.do_sequence()

    except Exception as e:
        pdb.set_trace()

    # daq.stop_acquistion()

    # close the window and file
    # win.close()
    # file.close()

    return (daq, file_save)


def get_task_info():
    ele_sti = [
        ['B', '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '-', 'L'],
        ['E', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['W', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z'],
        ['Q', '+', 'F', 'G', 'E', '-', 'S', 'Q', 'W', 'E', '<', 'A', 'Z']]
    color_sti = [['green', 'red', 'white', 'white', 'white', 'white', 'white',
                  'white', 'white', 'white', 'white', 'white', 'white']] * 4
    task_text = ['1/100', '2/100', '3/100', '4/100']
    task_color = [['white'], ['white'], ['white'], ['white']]
    time_flash = .25
    time_target = 2
    time_cross = .6
    timing_sti = [[time_target] + [time_cross] + [time_flash] * (len(ele_sti[0]) - 1)] * 4

    return (task_text, task_color, ele_sti, timing_sti, color_sti)
