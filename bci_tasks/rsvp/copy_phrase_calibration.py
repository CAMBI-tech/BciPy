# Calibration Task for RSVP

from __future__ import division
from psychopy import core, clock

from display.rsvp.rsvp_disp_modes import CopyPhraseTask
from helpers.acquisition_related import init_eeg_acquisition

from helpers.triggers import _write_triggers_from_sequence_copy_phrase
from helpers.stim_gen import target_rsvp_sequence_generator, get_task_info

from helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, get_user_input,
    trial_complete_message)


def rsvp_copy_phrase_calibration_task(win, parameters,
                                      file_save, fake):

    # Initialize Experiment clocks etc.
    frame_rate = win.getActualFrameRate()
    static_clock = core.StaticPeriod(screenHz=frame_rate)
    buffer_val = float(parameters['task_buffer_len']['value'])

    # Get alphabet for experiment
    alp = alphabet(parameters)

    # Start acquiring data
    try:
        experiment_clock = clock.Clock()
        # Initialize EEG Acquisition
        daq, server = init_eeg_acquisition(
            parameters, file_save, clock=experiment_clock, server=fake)
    except Exception as e:
        print "Data acquistion could not start!"
        raise e

    # Try Initializing the Copy Phrase Display Object
    try:
        rsvp = _init_copy_phrase_display_task(
            parameters, win, static_clock, experiment_clock)
    except Exception as e:
        raise e

    # Init Triggers
    trigger_save_location = file_save + '/triggers.txt'
    trigger_file = open(trigger_save_location, 'w')
    run = True

    # get the initial target letter
    copy_phrase = parameters['text_task']['value']
    target_letter = copy_phrase[0]
    text_task = '*'

    # check user input to make sure we should be going
    if not get_user_input(
            rsvp, parameters['wait_screen_message']['value'],
            parameters['wait_screen_message_color']['value'],
            first_run=True):
        run = False

    while run:

        # check user input to make sure we should be going
        if not get_user_input(
                rsvp, parameters['wait_screen_message']['value'],
                parameters['wait_screen_message_color']['value']):
            break

        # Try getting random sequence information given stimuli parameters
        try:
            # to-do implement color from params
            (ele_sti, timing_sti, color_sti) = target_rsvp_sequence_generator(
                alp, target_letter, parameters,
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
            print e
            raise e

        # Try executing the sequences
        try:
            rsvp.update_task_state(text=text_task, color_list=['white'])
            rsvp.draw_static()
            win.flip()

            # update task state
            rsvp.stim_sequence = ele_sti[0]
            # rsvp.text_task = text_task
            if parameters['is_txt_sti']['value']:
                rsvp.color_list_sti = color_sti[0]

            rsvp.time_list_sti = timing_sti[0]

            core.wait(buffer_val)
            sequence_timing = rsvp.do_sequence()

            _write_triggers_from_sequence_copy_phrase(
                sequence_timing,
                trigger_file,
                copy_phrase,
                text_task)

            core.wait(buffer_val)

            (target_letter, text_task, run) = fake_copy_phrase_decision(
                copy_phrase, target_letter, text_task)

        except Exception as e:
            raise e

    # update the final task state and say goodbye
    rsvp.update_task_state(text=text_task, color_list=['white'])
    rsvp.text = trial_complete_message(win, parameters)
    rsvp.draw_static()
    win.flip()

    core.wait(buffer_val)

    # Close this sessions trigger file and return some data
    trigger_file.close()

    # Wait some time before exiting so there is trailing eeg data saved
    core.wait(int(parameters['eeg_buffer_len']['value']))

    try:
        daq.stop_acquisition()
    except:
        # if not started, we can pass!
        pass

    if server:
        server.stop()

    return file_save


def _init_copy_phrase_display_task(
        parameters, win, static_clock, experiment_clock):
    rsvp = CopyPhraseTask(
        window=win, clock=static_clock,
        experiment_clock=experiment_clock,
        text_info=parameters['text_text']['value'],
        static_text_task=parameters['text_task']['value'],
        text_task='****',
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
        tr_pos_bg=(float(parameters['tr_pos_bg_x']['value']),
                   float(parameters['tr_pos_bg_y']['value'])),
        bl_pos_bg=(float(parameters['bl_pos_bg_x']['value']),
                   float(parameters['bl_pos_bg_y']['value'])),
        size_domain_bg=int(parameters['size_domain_bg']['value']),
        color_bg_txt=parameters['color_bg_txt']['value'],
        font_bg_txt=parameters['font_bg_txt']['value'],
        color_bar_bg=parameters['color_bar_bg']['value'],
        is_txt_sti=True if parameters[
            'is_txt_sti']['value'] == 'true' else False)

    return rsvp
