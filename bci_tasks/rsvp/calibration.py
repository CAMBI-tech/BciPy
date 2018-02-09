# Calibration Task for RSVP

from __future__ import division, print_function
from psychopy import prefs
prefs.general['audioLib'] = ['pygame']


from psychopy import core, sound
import time

from display.rsvp.rsvp_disp_modes import CalibrationTask
from helpers.acquisition_related import init_eeg_acquisition

from helpers.triggers import _write_triggers_from_sequence_calibration
from helpers.stim_gen import random_rsvp_calibration_seq_gen, get_task_info
from helpers.bci_task_related import (
    alphabet, trial_complete_message, get_user_input)


def rsvp_calibration_task(win, parameters, file_save, fake):
    '''RSVP Calibration Task

        Calibration task performs an RSVP stimulus sequence
            to elicit an ERP. Parameters will change how many stim
            and for how long they present. Parameters also change
            color and text / image inputs. 

        A task begins setting up variables --> initializing eeg -->
            awaiting user input to start -->
            setting up stimuli --> presenting sequences -->
            saving data

        Input: 
            win (PsychoPy Display Object)
            parameters (Dictionary)
            file_save (String)
            fake (Boolean)

        Output:
            file_save (String)

    '''



    # Initialize needed experiment information 
    frame_rate = win.getActualFrameRate()
    static_clock = core.StaticPeriod(screenHz=frame_rate)
    buffer_val = float(parameters['task_buffer_len']['value'])
    alp = alphabet(parameters)

    # Start acquiring data
    try:
        experiment_clock = core.Clock()

        # Initialize EEG Acquisition
        daq, server = init_eeg_acquisition(
            parameters, file_save, clock=experiment_clock, server=fake)

    except Exception as e:
        print("EEG initializing failed")
        raise e

    # Try initializing the calibration task
    try:
        rsvp = init_calibration_display_task(
            parameters, win, static_clock, experiment_clock)
    except Exception as e:
        raise e

    # Init Task Triggers and Run
    trigger_save_location = file_save + '/triggers.txt'
    trigger_file = open(trigger_save_location, 'w')
    run = True

    # Check user input to make sure we should be going
    if not get_user_input(rsvp, parameters['wait_screen_message']['value'],
                          parameters['wait_screen_message_color']['value'],
                          first_run=True):
        run = False

    # Begin the Experiment
    while run:

        # Get random sequence information given stimuli parameters
        try:
            (ele_sti, timing_sti,
             color_sti) = random_rsvp_calibration_seq_gen(
                alp, num_sti=int(parameters['num_sti']['value']),
                len_sti=int(parameters['len_sti']['value']), timing=[
                    float(parameters['time_target']['value']),
                    float(parameters['time_cross']['value']),
                    float(parameters['time_flash']['value'])],
                is_txt=rsvp.is_txt_sti,
                color=[
                    parameters['target_letter_color']['value'],
                    parameters['fixation_color']['value'],
                    parameters['stimuli_color']['value']])

            (task_text, task_color) = get_task_info(
                int(parameters['num_sti']['value']),
                parameters['task_color']['value'])

        # Catch the exception here if needed.
        except Exception as e:
            raise e

        # Execute the RSVP sequences
        try:
            for idx_o in range(len(task_text)):

                # check user input to make sure we should be going
                if not get_user_input(
                        rsvp, parameters['wait_screen_message']['value'],
                        parameters['wait_screen_message_color']['value']):
                    break

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
                rsvp.stim_sequence = ele_sti[idx_o]

                # check if text stimuli or not for color information
                if parameters['is_txt_sti']['value']:
                    rsvp.color_list_sti = color_sti[idx_o]

                rsvp.time_list_sti = timing_sti[idx_o]

                # Wait for a time
                core.wait(buffer_val)

                # Do the sequence
                last_sequence_timing = rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, trigger_file)

                # Wait for a time
                core.wait(buffer_val)

            # Set run to False to stop looping
            run = False

        except Exception as e:
            raise e

    # Say Goodbye!
    rsvp.text = trial_complete_message(win, parameters)
    rsvp.draw_static()
    win.flip()

    # Give the system time to process
    core.wait(buffer_val)

    # Close this sessions trigger file and return some data
    trigger_file.close()

    # print offset.. maybe add to text file with ISI added via params?
    print(daq.offset)

    # Wait some time before exiting so there is trailing eeg data saved
    core.wait(int(parameters['eeg_buffer_len']['value']))

    try:
        daq.stop_acquisition()
    except Exception as e:
        raise e

    if server:
        server.stop()

    return file_save


def init_calibration_display_task(
        parameters, win, static_clock, experiment_clock):
    rsvp = CalibrationTask(
        window=win, clock=static_clock,
        experiment_clock=experiment_clock,
        text_info=parameters['text_text']['value'],
        color_info=parameters['color_text']['value'],
        pos_info=(float(parameters['pos_text_x']['value']),
                  float(parameters['pos_text_y']['value'])),
        height_info=float(parameters['txt_height']['value']),
        font_info=parameters['font_text']['value'],
        color_task=['white'],
        font_task=parameters['font_task']['value'],
        height_task=float(parameters['height_task']['value']),
        font_sti=parameters['font_sti']['value'],
        pos_sti=(float(parameters['pos_sti_x']['value']),
                 float(parameters['pos_sti_y']['value'])),
        sti_height=float(parameters['sti_height']['value']),
        stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
        time_list_sti=[3] * 10,
        size_domain_bg=int(parameters['size_domain_bg']['value']),
        color_bg_txt=parameters['color_bg_txt']['value'],
        font_bg_txt=parameters['font_bg_txt']['value'],
        color_bar_bg=parameters['color_bar_bg']['value'],
        is_txt_sti=True if parameters['is_txt_sti']['value'] == 'true' else False)
    return rsvp
