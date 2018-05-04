# Calibration Task for RSVP

from psychopy import core
from bci_tasks.task import Task

from display.rsvp.rsvp_disp_modes import CopyPhraseDisplay

from helpers.triggers import _write_triggers_from_sequence_copy_phrase
from helpers.save import _save_session_related_data
from helpers.eeg_model_wrapper import CopyPhraseWrapper

from helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, process_data_for_decision,
    trial_complete_message, get_user_input)


class RSVPCopyPhraseTask(Task):
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
            trained signal_model, loaded before session started
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """

    def __init__(
            self, win, daq, parameters, file_save, classifier, lmodel, fake):

        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = float(parameters['task_buffer_len']['value'])
        self.alp = alphabet(parameters)
        self.rsvp = _init_copy_phrase_display_task(
            self.parameters, self.window,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save
        trigger_save_location = self.file_save + '/triggers.txt'
        self.session_save_location = self.file_save + '/session.json'
        self.trigger_file = open(trigger_save_location, 'w')

        self.wait_screen_message = parameters['wait_screen_message']['value']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']['value']

        self.num_sti = int(parameters['num_sti']['value'])
        self.len_sti = int(parameters['len_sti']['value'])
        self.timing = [float(parameters['time_target']['value']),
                       float(parameters['time_cross']['value']),
                       float(parameters['time_flash']['value'])]

        self.color = [parameters['target_letter_color']['value'],
                      parameters['fixation_color']['value'],
                      parameters['stimuli_color']['value']]

        self.task_info_color = parameters['task_color']['value']

        self.stimuli_height = float(parameters['sti_height']['value'])

        self.is_txt_sti = True if parameters['is_txt_sti']['value'] == 'true' \
            else False,
        self.eeg_buffer = int(parameters['eeg_buffer_len']['value'])
        self.copy_phrase = parameters['text_task']['value']

        self.max_seq_length = int(parameters['max_seq_len']['value'])
        self.fake = fake
        self.lmodel = lmodel
        self.classifier = classifier

    def execute(self):
        text_task = str(self.copy_phrase[0:int(len(self.copy_phrase) / 2)])
        task_list = [(str(self.copy_phrase),
                      str(self.copy_phrase[0:int(len(self.copy_phrase) / 2)]))]

        # Try Initializing Copy Phrase Wrapper:
        #       (sig_pro, decision maker, signal_model)
        try:
            copy_phrase_task = CopyPhraseWrapper(self.classifier, self.daq._device.fs,
                                                 2, self.alp, task_list=task_list,
                                                 lmodel=self.lmodel,
                                                 is_txt_sti=self.is_txt_sti)
        except Exception as e:
            print("Error initializing Copy Phrase Task")
            raise e

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
            'session': self.file_save,
            'session_type': 'Copy Phrase',
            'paradigm': 'RSVP',
            'epochs': {},
            'total_time_spent': self.experiment_clock.getTime(),
            'total_number_epochs': 0,
        }

        # Save session data
        _save_session_related_data(self.session_save_location, data)

        # check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        # Start the Session!
        while run:

            # check user input to make sure we should be going
            if not get_user_input(self.rsvp, self.wait_screen_message,
                                  self.wait_screen_message_color):
                break

            # Why bs for else? #changeforrelease
            if self.copy_phrase[0:len(text_task)] == text_task:
                target_letter = self.copy_phrase[len(text_task)]
            else:
                target_letter = '<'

            # Get sequence information
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

            # Update task state and reset the static
            self.rsvp.update_task_state(text=text_task, color_list=['white'])
            self.rsvp.draw_static()
            self.window.flip()

            # Setup the new Stimuli
            self.rsvp.stim_sequence = ele_sti[0]
            if self.is_txt_sti:
                self.rsvp.color_list_sti = color_sti[0]
            self.rsvp.time_list_sti = timing_sti[0]

            # Pause for a time
            core.wait(self.buffer_val)

            # Do the self.RSVP sequence!
            sequence_timing = self.rsvp.do_sequence()

            # Write triggers to file
            _write_triggers_from_sequence_copy_phrase(
                sequence_timing,
                self.trigger_file,
                self.copy_phrase,
                text_task)

            core.wait(self.buffer_val)

            # reshape the data and triggers as needed for later modules
            raw_data, triggers, target_info = \
                process_data_for_decision(sequence_timing, self.daq)

            # Uncomment this to turn off fake decisions, but use fake data.
            # fake = False
            if self.fake:
                # Construct Data Record
                data['epochs'][epoch_counter][epoch_index] = {
                    'stimuli': ele_sti,
                    'eeg_len': len(raw_data),
                    'timing_sti': timing_sti,
                    'triggers': triggers,
                    'target_info': target_info,
                    'target_letter': target_letter,
                    'current_text': text_task,
                    'copy_phrase': self.copy_phrase}

                # Evaulate this sequence
                (target_letter, text_task, run) = \
                    fake_copy_phrase_decision(self.copy_phrase,
                                              target_letter,
                                              text_task)
                new_epoch = True
                # Update next state for this record
                data['epochs'][
                    epoch_counter][
                    epoch_index][
                    'next_display_state'] = \
                    text_task

            else:
                # Evaluate this sequence, returning whether to gen a new
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
                    'copy_phrase': self.copy_phrase,
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

            # Update time spent and save data
            data['total_time_spent'] = self.experiment_clock.getTime()
            data['total_number_epochs'] = epoch_counter
            _save_session_related_data(self.session_save_location, data)

            # Decide whether to keep the task going
            if (text_task == self.copy_phrase or
                    seq_counter > self.max_seq_length):

                run = False

            # Increment sequence counter
            seq_counter += 1

        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self.buffer_val)

        if self.daq.is_calibrated:
            _write_triggers_from_sequence_copy_phrase(
                ['offset', self.daq.offset], self.trigger_file,
                self.copy_phrase, text_task, offset=True)

        # Close the trigger file for this session
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

        return self.file_save

    def name(self):
        return 'RSVP Copy Phrase Task'


def _init_copy_phrase_display_task(
        parameters, win, static_clock, experiment_clock):
    rsvp = CopyPhraseDisplay(
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
        is_txt_sti=True if parameters['is_txt_sti']['value'] == 'true' else False,
        trigger_type=parameters['trigger_type']['value'])

    return rsvp
