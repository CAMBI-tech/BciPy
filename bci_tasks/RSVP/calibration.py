# Calibration Task for RSVP

from __future__ import division
from psychopy import core, event

from display.rsvp_disp_modes import CalibrationTask

from helpers.triggers import _write_triggers_from_sequence_calibration
from helpers.stim_gen import random_rsvp_calibration_seq_gen, get_task_info
from helpers.bci_task_related import alphabet


def rsvp_calibration_task(win, daq, parameters, file_save):
    # Initialize Experiment clocks etc.
    frame_rate = win.getActualFrameRate()
    clock = core.StaticPeriod(screenHz=frame_rate)
    experiment_clock = core.MonotonicClock(start_time=None)

    # Get alphabet for experiment
    alp = alphabet()

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
        # check user input to make sure we should be going
        keys = event.getKeys(keyList=['space', 'escape'])

        if keys:
            # pause?
            if keys[0] == 'space':
                event.waitKeys(keyList=["space"])

            # escape?
            if keys[0] == 'escape':
                break

        # Try getting random sequence information given stimuli parameters
        try:
            (ele_sti, timing_sti,
             color_sti) = random_rsvp_calibration_seq_gen(
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
                core.wait(float(parameters['task_buffer_len']['value']))

                # Do the sequence
                last_sequence_timing = rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, trigger_file)

                # Wait for a time
                core.wait(float(parameters['task_buffer_len']['value']))

            # Set run to False to stop looping
            run = False

        except Exception as e:
            raise e

    # Close this sessions trigger file and return some data
    trigger_file.close()

    # Wait some time before exiting so there is trailing eeg data saved
    core.wait(int(parameters['eeg_buffer_len']['value']))

    return file_save
