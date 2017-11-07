# Calibration Task for RSVP

from __future__ import division
from psychopy import core

from display.rsvp_disp_modes import CopyPhraseTask
from helpers.trigger_helpers import _write_triggers_from_sequence_copy_phrase
from helpers.stim_gen import (
    random_rsvp_sequence_generator, get_task_info, rsvp_copy_phrase_generator)


alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']


def rsvp_copy_phrase_task(win, daq, parameters, file_save):

    # Initialize Experiment clocks etc.
    frame_rate = win.getActualFrameRate()
    clock = core.StaticPeriod(screenHz=frame_rate)
    experiment_clock = core.MonotonicClock(start_time=None)

    # Start acquiring data
    try:
        daq.clock = experiment_clock
        daq.start_acquisition()
    except Exception as e:
        print "Data acquistion could not start!"
        raise e

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
            tr_pos_bg=(float(parameters['tr_pos_bg_x']['value']),
                       float(parameters['tr_pos_bg_y']['value'])),
            bl_pos_bg=(float(parameters['bl_pos_bg_x']['value']),
                       float(parameters['bl_pos_bg_y']['value'])),
            size_domain_bg=int(parameters['size_domain_bg']['value']),
            color_bg_txt=parameters['color_bg_txt']['value'],
            font_bg_txt=parameters['font_bg_txt']['value'],
            color_bar_bg=parameters['color_bar_bg']['value'],
            is_txt_sti=parameters['is_txt_sti']['value'])
    except Exception as e:
        raise e

    # Init Triggers
    trigger_save_location = file_save + '/triggers.txt'
    trigger_file = open(trigger_save_location, 'w')
    run = True

    while run is True:
        # [to-do] allow pausing and exiting. See psychopy getKeys()

        # Try getting random sequence information given stimuli parameters
        try:
            (ele_sti, timing_sti, color_sti) = rsvp_copy_phrase_generator(
                alp, 'C', num_sti=int(parameters['num_sti']['value']),
                len_sti=int(parameters['len_sti']['value']), timing=[
                    float(parameters['time_target']['value']),
                    float(parameters['time_cross']['value']),
                    float(parameters['time_flash']['value'])])

            (task_text, task_color) = get_task_info(
                int(parameters['num_sti']['value']),
                parameters['task_color']['value'])

        # Catch the exception here if needed.
        except Exception as e:
            print e
            raise e

        # Try executing the sequences
        try:
            counter = 0
            for idx_o in range(len(task_text)):

                # update task state
                rsvp.ele_list_sti = ele_sti[counter]
                if parameters['is_txt_sti']['value']:
                    rsvp.color_list_sti = color_sti[counter]

                rsvp.time_list_sti = timing_sti[counter]

                #
                core.wait(.4)
                sequence_timing = rsvp.do_sequence()

                # _write_triggers_from_sequence_copy_phrase(
                #     sequence_timing,
                #     file,
                #     parameters['text_task']['value'],
                #     task_text[idx_o])

                # # Get parameters from Bar Graph and schedule
                # rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[counter],
                #                     weight=dummy_bar_schedule_p[counter])

                core.wait(.5)
                # if show_bg:
                #     rsvp.show_bar_graph()

                counter += 1

            # Set run to False to stop looping
            run = False

        except Exception as e:
            print e
            raise e

    # Close this sessions trigger file and return some data
    trigger_file.close()

    return file_save
