# Calibration Task for RSVP

from __future__ import division
from psychopy import core
import numpy as np

from display.rsvp_disp_modes import CopyPhraseTask
from helpers.triggers import _write_triggers_from_sequence_copy_phrase
from helpers.stim_gen import rsvp_copy_phrase_seq_generator
from bci_tasks.wrappers import CopyPhraseWrapper

from helpers.bci_task_related import fake_copy_phrase_decision, alphabet


def rsvp_copy_phrase_task(win, daq, parameters, file_save, classifier,
                          fake=False):
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
    text_task = str(copy_phrase[0:int(len(copy_phrase) / 2)])

    # Init Copy Phrase
    # TODO: get model, fs and k from .pkl besides model

    task_list = [(str(copy_phrase), str(copy_phrase[0:int(len(copy_phrase) /
                                                          2)]))]
    try:
        copy_phrase_task = CopyPhraseWrapper(classifier, 300, 2, alp,
                                             task_list=task_list)
    except Exception as e:
        print "Error initializing Copy Phrase Task"

    # Set new epoch to true and sequence counter to zero
    new_epoch = True
    seq_counter = 0

    # Start the experiment!
    while run is True:
        # [to-do] allow pausing and exiting. See psychopy getKeys()

        # Why bs for else?
        if copy_phrase[0:len(text_task)] == text_task:
            target_letter = copy_phrase[len(text_task)]
        else:
            target_letter = '<'

        # Try getting sequence information
        try:
            if new_epoch:

                new_epoch, sti = copy_phrase_task.initialize_epoch()
                ele_sti = sti[0]
                timing_sti = sti[1]
                color_sti = sti[2]

        # Catch the exception here if needed.
        except Exception as e:
            print "Error Initializing Epoch!"
            raise e

        # Try executing the given sequences. This is where display is used!
        try:

            # Update task state and reset the static
            rsvp.update_task_state(text=text_task, color_list=['white'])
            rsvp.draw_static()
            win.flip()

            # Setup the new Stimuli
            rsvp.ele_list_sti = ele_sti[0]
            if parameters['is_txt_sti']['value']:
                rsvp.color_list_sti = color_sti[0]
            rsvp.time_list_sti = timing_sti[0]

            # Pause for a time
            core.wait(.5)

            # Do the RSVP sequence!
            sequence_timing = rsvp.do_sequence()

            # Write triggers to file
            _write_triggers_from_sequence_copy_phrase(
                sequence_timing,
                trigger_file,
                copy_phrase,
                text_task)

            # Get timing of the first and last stimuli
            _, first_stim_time = sequence_timing[0]
            _, last_stim_time = sequence_timing[len(sequence_timing) - 1]

            # define my first and last time points
            time1 = first_stim_time
            time2 = first_stim_time + last_stim_time + 2

            # Construct triggers to send off for processing
            triggers = [(text, timing - time1)
                        for text, timing in sequence_timing]

            # Query for raw data
            try:
                raw_data = daq.get_data(start=time1, end=time2)

                # TODO: We hardcoded 0 as it is the data location
                raw_data = np.array([np.array(raw_data[i][0]) for i in
                                     range(len(raw_data))]).transpose()

            except Exception as e:
                print "Error in daq get_data()"
                raise e

            # # Get parameters from Bar Graph and schedule
            # rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[counter],
            #                     weight=dummy_bar_schedule_p[counter])

            core.wait(.5)
            # if show_bg:
            #     rsvp.show_bar_graph()

            # TODO: Don't forget you sinned
            fake = False
            try:
                if fake:
                    (target_letter, text_task, run) = fake_copy_phrase_decision(
                        copy_phrase, target_letter, text_task)
                else:
                    print(raw_data.shape)
                    target_info = ['nontarget'] * len(triggers)
                    new_epoch, sti = \
                        copy_phrase_task.evaluate_sequence(raw_data, triggers,
                                                           target_info)
                    ele_sti = sti[0]
                    timing_sti = sti[1]
                    color_sti = sti[2]
                    text_task = copy_phrase_task.decision_maker.displayed_state
            except Exception as e:
                raise e

        except Exception as e:
            raise e

        run = (copy_phrase_task.decision_maker.displayed_state !=
               copy_phrase and seq_counter < 50)
        seq_counter += 1

    # Close the trigger file for this session
    trigger_file.close()

    # Wait some time before exiting so there is trailing eeg data saved
    core.wait(int(parameters['eeg_buffer_len']['value']))

    return file_save
