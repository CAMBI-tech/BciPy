# Calibration Task for RSVP

from __future__ import division
from psychopy import core

from display.rsvp_disp_modes import CalibrationTask
from helpers.trigger_helpers import _write_triggers_from_sequence_calibration
from helpers.stim_gen import random_rsvp_sequence_generator, get_task_info

alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']


def rsvp_calibration_task(win, daq, parameters, file_save):

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

    # Try running the calibration task
    try:
        rsvp = CalibrationTask(
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
        raise e

    # Init Task
    trigger_save_location = file_save + '/triggers.txt'
    trigger_file = open(trigger_save_location, 'w')
    run = True

    while run is True:
        # [to-do] allow pausing and exiting. See psychopy getKeys()

        # Try getting random sequence information given stimuli parameters
        try:
            (ele_sti, timing_sti, color_sti) = random_rsvp_sequence_generator(
                alp, num_sti=int(parameters['num_sti']['value']),
                len_sti=int(parameters['len_sti']['value']), timing=[
                    float(parameters['time_target']['value']),
                    float(parameters['time_cross']['value']),
                    float(parameters['time_flash']['value'])])

            (task_text, task_color) = get_task_info(
                int(parameters['num_sti']['value']),
                parameters['task_color']['value'])

        # Catch the exception here if needed.
        except Exception as e:
            raise e

        # Try executing the sequences
        try:
            for idx_o in range(len(task_text)):

                # update task state
                rsvp.update_task_state(
                    text=task_text[idx_o],
                    color_list=task_color[idx_o])

                # Draw and flip screen
                rsvp.draw_static()
                win.flip()

                # Get height
                rsvp.sti.height = float(parameters['sti_height']['value'])

                # Schedule a sequence
                rsvp.ele_list_sti = ele_sti[idx_o]

                # check if text stimuli or not for color information
                if parameters['is_txt_sti']['value']:
                    rsvp.color_list_sti = color_sti[idx_o]

                rsvp.time_list_sti = timing_sti[idx_o]

                # Wait for a time
                core.wait(.4)

                # Do the sequence
                last_sequence_timing = rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, trigger_file)

                # Wait for a time
                core.wait(.5)

            # Set run to False to stop looping
            run = False

        except Exception as e:
            raise e

    # Close this sessions trigger file and return some data
    trigger_file.close()
    return file_save
