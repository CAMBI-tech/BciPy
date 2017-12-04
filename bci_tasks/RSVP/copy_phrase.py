# Calibration Task for RSVP

from __future__ import division
from psychopy import core

from display.rsvp_disp_modes import CopyPhraseTask
from helpers.trigger_helpers import _write_triggers_from_sequence_copy_phrase
from helpers.stim_gen import rsvp_copy_phrase_seq_generator


alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']


def rsvp_copy_phrase_task(win, daq, parameters, file_save, fake=True):

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
            static_text_task=parameters['text_task']['value'],
            text_task='****',
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

    # get the initial target letter
    copy_phrase = parameters['text_task']['value']
    target_letter = copy_phrase[0]
    text_task = '*'

    while run is True:
        # [to-do] allow pausing and exiting. See psychopy getKeys()

        # Try getting random sequence information given stimuli parameters
        try:
            # to-do implement color from params
            (ele_sti, timing_sti, color_sti) = rsvp_copy_phrase_seq_generator(
                alp, target_letter,
                len_sti=int(parameters['len_sti']['value']), timing=[
                    float(parameters['time_target']['value']),
                    float(parameters['time_cross']['value']),
                    float(parameters['time_flash']['value'])])

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
            rsvp.ele_list_sti = ele_sti[0]
            # rsvp.text_task = text_task
            if parameters['is_txt_sti']['value']:
                rsvp.color_list_sti = color_sti[0]

            rsvp.time_list_sti = timing_sti[0]

            core.wait(.4)
            sequence_timing = rsvp.do_sequence()

            _write_triggers_from_sequence_copy_phrase(
                sequence_timing,
                trigger_file,
                copy_phrase,
                text_task)

            # Get timing of the first and last stimuli
            _, first_stim_time = sequence_timing[0]
            _, last_stim_time = sequence_timing[len(sequence_timing) - 1]

            # define my first and last time points
            time1 = first_stim_time - .5
            time2 = last_stim_time + .5

            # Construct triggers
            triggers = [(text, timing - time1)
                        for text, timing in sequence_timing]

            # Query for raw data
            try:
                raw_data = daq.get_data(time1, time2)
            except Exception as e:
                print "error in get_data()"
                raise e

            # # Get parameters from Bar Graph and schedule
            # rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[counter],
            #                     weight=dummy_bar_schedule_p[counter])

            core.wait(.5)
            # if show_bg:
            #     rsvp.show_bar_graph()

            if fake:
                (target_letter, text_task, run) = fake_copy_phrase_decision(
                    copy_phrase, target_letter, text_task)
            else:
                raise Exception('Real decision maker not implemented yet')

        except Exception as e:
            print e
            raise e

    # Close this sessions trigger file and return some data
    trigger_file.close()

    # Wait some time before exiting so there is trailing eeg data saved
    core.wait(int(parameters['eeg_buffer_len']['value']))

    return file_save


def fake_copy_phrase_decision(copy_phrase, target_letter, text_task):
    if text_task is '*':
        length_of_spelled_letters = 0
    else:
        length_of_spelled_letters = len(text_task)

    length_of_phrase = len(copy_phrase)

    if length_of_spelled_letters is 0:
        text_task = copy_phrase[length_of_spelled_letters]
    else:
        text_task += copy_phrase[length_of_spelled_letters]

    length_of_spelled_letters += 1

    # If there is still text to be spelled, update the text_task
    # and target letter
    if length_of_spelled_letters < length_of_phrase:
        next_target_letter = copy_phrase[length_of_spelled_letters]

        run = True

    # else, end the run
    else:
        run = False
        next_target_letter = None
        text_task = None

    return next_target_letter, text_task, run
