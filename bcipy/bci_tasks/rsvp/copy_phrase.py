from psychopy import core
from bcipy.bci_tasks.task import Task

from bcipy.display.rsvp.rsvp_disp_modes import CopyPhraseDisplay

from bcipy.helpers.triggers import _write_triggers_from_sequence_copy_phrase
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.eeg_model_related import CopyPhraseWrapper

from bcipy.helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, process_data_for_decision,
    trial_complete_message, get_user_input)
import logging


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
        self.buffer_val = parameters['task_buffer_len']
        self.alp = alphabet(parameters)
        self.rsvp = _init_copy_phrase_display(
            self.parameters, self.window, self.daq,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save

        trigger_save_location = f"{self.file_save}/{parameters['triggers_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w')
        self.session_save_location = f"{self.file_save}/{parameters['session_file_name']}"

        self.wait_screen_message = parameters['wait_screen_message']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']

        self.num_sti = parameters['num_sti']
        self.len_sti = parameters['len_sti']
        self.timing = [parameters['time_target'],
                       parameters['time_cross'],
                       parameters['time_flash']]

        self.color = [parameters['target_letter_color'],
                      parameters['fixation_color'],
                      parameters['stimuli_color']]

        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['sti_height']

        self.is_txt_sti = parameters['is_txt_sti']
        self.eeg_buffer = parameters['eeg_buffer_len']
        self.copy_phrase = parameters['text_task']
        self.spelled_letters_count = int(
            parameters['spelled_letters_count'])
        if self.spelled_letters_count > len(self.copy_phrase):
            logging.debug("Already spelled letters exceeds phrase length.")
            self.spelled_letters_count = 0

        self.max_seq_length = parameters['max_seq_len']
        self.max_seconds = parameters['max_minutes'] * 60  # convert to seconds
        self.fake = fake
        self.lmodel = lmodel
        self.classifier = classifier
        self.down_sample_rate = parameters['down_sampling_rate']

    def execute(self):
        text_task = str(self.copy_phrase[0:self.spelled_letters_count])
        task_list = [(str(self.copy_phrase),
                      str(self.copy_phrase[0:self.spelled_letters_count]))]

        # Try Initializing Copy Phrase Wrapper:
        #       (sig_pro, decision maker, signal_model)
        try:
            copy_phrase_task = CopyPhraseWrapper(signal_model=self.classifier, fs=self.daq.device_info.fs,
                                                 k=2, alp=self.alp, task_list=task_list,
                                                 lmodel=self.lmodel,
                                                 is_txt_sti=self.is_txt_sti,
                                                 device_name=self.daq.device_info.name,
                                                 device_channels=self.daq.device_info.channels)
        except Exception as e:
            print("Error initializing Copy Phrase Task")
            raise e

        # Set new epoch (whether to present a new epoch),
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

            if seq_counter == 0:
                del sequence_timing[0]

            # reshape the data and triggers as needed for later modules
            raw_data, triggers, target_info = \
                process_data_for_decision(sequence_timing, self.daq)

            # Uncomment this to turn off fake decisions, but use fake data.
            # self.fake = False
            if not self.fake:
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

                # Evaluate this sequence
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
            max_tries_exceeded = seq_counter >= self.max_seq_length
            max_time_exceeded = data['total_time_spent'] >= self.max_seconds
            if (text_task == self.copy_phrase or max_tries_exceeded or
                    max_time_exceeded):
                if max_tries_exceeded:
                    logging.debug("Max tries exceeded: to allow for more tries"
                                  " adjust the Maximum Sequence Length "
                                  "(max_seq_len) parameter.")
                if max_time_exceeded:
                    logging.debug("Max time exceeded. To allow for more time "
                                  "adjust the max_minutes parameter.")
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


def _init_copy_phrase_display(
        parameters, win, daq, static_clock, experiment_clock):
    rsvp = CopyPhraseDisplay(
        window=win, clock=static_clock,
        experiment_clock=experiment_clock,
        marker_writer=daq.marker_writer,
        text_info=parameters['text_text'],
        static_text_task=parameters['text_task'],
        text_task='****',
        color_info=parameters['color_text'],
        pos_info=(parameters['pos_text_x'],
                  parameters['pos_text_y']),
        height_info=parameters['txt_height'],
        font_info=parameters['font_text'],
        color_task=['white'],
        font_task=parameters['font_task'],
        height_task=parameters['height_task'],
        font_sti=parameters['font_sti'],
        pos_sti=(parameters['pos_sti_x'],
                 parameters['pos_sti_y']),
        sti_height=parameters['sti_height'],
        stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
        time_list_sti=[3] * 10,
        tr_pos_bg=(parameters['tr_pos_bg_x'],
                   parameters['tr_pos_bg_y']),
        bl_pos_bg=(parameters['bl_pos_bg_x'],
                   parameters['bl_pos_bg_y']),
        size_domain_bg=parameters['size_domain_bg'],
        color_bg_txt=parameters['color_bg_txt'],
        font_bg_txt=parameters['font_bg_txt'],
        color_bar_bg=parameters['color_bar_bg'],
        is_txt_sti=parameters['is_txt_sti'],
        trigger_type=parameters['trigger_type'])

    return rsvp
