# Calibration Task for RSVP

from __future__ import division
from psychopy import core, event
import numpy as np

from display.rsvp_disp_modes import CopyPhraseTask
from helpers.triggers import _write_triggers_from_sequence_copy_phrase
from helpers.save import _save_session_related_data
from helpers.eeg_model_wrapper import CopyPhraseWrapper

from helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, _process_data_for_decision,
    trial_complete_message)


def rsvp_copy_phrase_task(win, daq, parameters, file_save, classifier,
                          lmodel=None,
                          fake=False):
    """RSVP Copy Phrase Task.

    Initializes and runs all needed code for executing a copy phrase task. A
        phrase is set in parameters and necessary objects (eeg, display) are
        passed to this function. Certain Wrappers and Task Specific objects are
        executed here.

    Parameters
    ----------
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        daq : object,
            data acquisition object initialized for the desired protocol
        file_save : str,
            path location of where to save data from the session
        classifier : loaded pickle file,
            trained eeg_model, loaded before session started
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """

    # Initialize Experiment clocks etc.
    frame_rate = win.getActualFrameRate()
    clock = core.StaticPeriod(screenHz=frame_rate)
    buffer_val = float(parameters['task_buffer_len']['value'])

    # Get alphabet for experiment
    alp = alphabet()

    # Start acquiring data and set the experiment clock
    try:
        experiment_clock = core.MonotonicClock(start_time=None)
        daq._clock = experiment_clock
        daq.start_acquisition()
    except Exception as e:
        print "Data acquistion could not start!"
        raise e

    # Try Initializing the Copy Phrase Display Object
    try:
        rsvp = _init_copy_phrase_display_task(
            parameters, win, clock, experiment_clock)
    except Exception as e:
        raise e

    # Init Triggers
    trigger_save_location = file_save + '/triggers.txt'
    trigger_file = open(trigger_save_location, 'w')

    # Init Session File
    session_save_location = file_save + '/session.json'

    # Get the copy_phrase, initial targets and task list
    copy_phrase = parameters['text_task']['value']
    text_task = str(copy_phrase[0:int(len(copy_phrase) / 2)])
    # TODO: get model, fs and k from .pkl besides model. #changeforrelease
    task_list = [(str(copy_phrase), str(copy_phrase[0:int(len(copy_phrase) /
                                                          2)]))]

    # Try Initializing Copy Phrase Wrapper:
    #       (sig_pro, decision maker, eeg_model)
    try:
        copy_phrase_task = CopyPhraseWrapper(classifier, daq._device.fs,
                                             2, alp, task_list=task_list,
                                             lmodel=lmodel)
    except Exception as e:
        print "Error initializing Copy Phrase Task"

    # Set new epoch (wheter to present a new epoch),
    #   run (whether to cont. session),
    #   sequence counter (how many seq have occured).
    #   epoch counter and index (what epoch, and how many sequences within it)
    new_epoch = True
    run = True
    seq_counter = 0
    epoch_counter = 0
    epoch_index = 0

    # Init session data and save before beginning
    data = {
        'session': file_save,
        'session_type': 'Copy Phrase',
        'paradigm': 'RSVP',
        'epochs': {},
        'total_time_spent': experiment_clock.getTime(),
        'total_number_epochs': 0,
    }

    # Save session data
    _save_session_related_data(session_save_location, data)

    # Start the Session!
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

        # Why bs for else? #changeforrelease
        if copy_phrase[0:len(text_task)] == text_task:
            target_letter = copy_phrase[len(text_task)]
        else:
            target_letter = '<'

        # Try getting sequence information
        try:
            if new_epoch:

                # Init an epoch, getting initial stimuli
                new_epoch, sti = copy_phrase_task.initialize_epoch()
                ele_sti = sti[0]
                timing_sti = sti[1]
                color_sti = sti[2]

                # Increase epoch number and reset epoch index
                epoch_counter += 1
                data['epochs'][epoch_counter] = {}
                epoch_index = 0
            else:
                epoch_index += 1

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
            core.wait(buffer_val)

            # Do the RSVP sequence!
            sequence_timing = rsvp.do_sequence()

            # Write triggers to file
            _write_triggers_from_sequence_copy_phrase(
                sequence_timing,
                trigger_file,
                copy_phrase,
                text_task)

            # # Get parameters from Bar Graph and schedule
            # rsvp.bg.schedule_to(letters=dummy_bar_schedule_t[counter],
            #                     weight=dummy_bar_schedule_p[counter])

            core.wait(buffer_val)
            # if show_bg:
            #     rsvp.show_bar_graph()

            try:

                # reshape the data and triggers as needed for later modules
                raw_data, triggers, target_info = \
                    _process_data_for_decision(sequence_timing, daq)

                # Uncomment this to turn off fake decisions, but use fake data.
                # fake = False
                if fake:
                    # Construct Data Record
                    data['epochs'][epoch_counter][epoch_index] = {
                        'stimuli': ele_sti,
                        'eeg_len': len(raw_data),
                        'timing_sti': timing_sti,
                        'triggers': triggers,
                        'target_info': target_info,
                        'target_letter': target_letter,
                        'current_text': text_task,
                        'copy_phrase': copy_phrase}

                    # Evaulate this sequence
                    (target_letter, text_task, run) = \
                        fake_copy_phrase_decision(copy_phrase, target_letter,
                                                  text_task)
                    new_epoch = True
                    # Update next state for this record
                    data['epochs'][
                        epoch_counter][
                        epoch_index][
                        'next_display_state'] = \
                        text_task

                else:
                    # Evaulate this sequence, returning wheter to gen a new
                    #  epoch (seq) or stimuli to present
                    new_epoch, sti = \
                        copy_phrase_task.evaluate_sequence(raw_data, triggers,
                                                           target_info)

                    # Construct Data Record
                    data['epochs'][epoch_counter][epoch_index] = {
                        'stimuli': ele_sti,
                        'eeg_len': len(raw_data),
                        'timing_sti': timing_sti,
                        'triggers': triggers,
                        'target_info': target_info,
                        'current_text': text_task,
                        'copy_phrase': copy_phrase,
                        'next_display_state':
                            copy_phrase_task.decision_maker.displayed_state,
                        'lm_evidence': copy_phrase_task
                            .conjugator
                            .evidence_history['LM'][0]
                            .tolist(),
                        'eeg_evidence': copy_phrase_task
                            .conjugator
                            .evidence_history['ERP'][0]
                            .tolist(),
                        'likelihood': copy_phrase_task
                            .conjugator.likelihood.tolist()
                    }

                    # If new_epoch is False, get the stimuli info returned
                    if not new_epoch:
                        ele_sti = sti[0]
                        timing_sti = sti[1]
                        color_sti = sti[2]

                    # Get the current task text from the decision maker
                    text_task = copy_phrase_task.decision_maker.displayed_state

            except Exception as e:
                raise e

        except Exception as e:
            raise e

        # Update time spent and save data
        data['total_time_spent'] = experiment_clock.getTime()
        data['total_number_epochs'] = epoch_counter
        _save_session_related_data(session_save_location, data)

        # Decide whether to keep the task going
        if (text_task == copy_phrase or
                seq_counter > int(parameters['max_seq_len']['value'])):

            # Update the final task state and say goodbye!
            rsvp.update_task_state(text=text_task, color_list=['white'])
            rsvp.text = trial_complete_message(win, parameters)
            rsvp.draw_static()
            win.flip()

            core.wait(buffer_val)

            break

        # Increment sequence counter
        seq_counter += 1

    # Let the user know stopping criteria was met and stop
    print "Stopping criteria met!"

    # Close the trigger file for this session
    trigger_file.close()

    # Wait some time before exiting so there is trailing eeg data saved
    core.wait(int(parameters['eeg_buffer_len']['value']))

    return file_save


def _init_copy_phrase_display_task(parameters, win, clock, experiment_clock):
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

    return rsvp
