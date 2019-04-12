from psychopy import core

from bcipy.tasks.task import Task
from bcipy.display.rsvp.mode.copy_phrase import CopyPhraseDisplay
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.triggers import _write_triggers_from_sequence_copy_phrase
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.signal_model import CopyPhraseWrapper
from bcipy.helpers.task import (
    fake_copy_phrase_decision, alphabet, process_data_for_decision,
    trial_complete_message, get_user_input)


class RSVPCopyPhraseTask(Task):
    """RSVP Copy Phrase Task.

    Initializes and runs all needed code for executing a copy phrase task. A
        phrase is set in parameters and necessary objects (daq, display) are
        passed to this function. Certain Wrappers and Task Specific objects are
        executed here.

    Parameters
    ----------
        win : object,
            display window to present visual stimuli.
        daq : object,
            data acquisition object initialized for the desired protocol
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        file_save : str,
            path location of where to save data from the session
        signal_model : loaded pickle file,
            trained signal model.
        language_model: object,
            trained language model.
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """

    TASK_NAME = 'RSVP Copy Phrase Task'

    def __init__(
            self, win, daq, parameters, file_save, signal_model, language_model, fake):
        super(RSVPCopyPhraseTask, self).__init__()

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

        trigger_save_location = f"{self.file_save}/{parameters['trigger_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w')
        self.session_save_location = f"{self.file_save}/{parameters['session_file_name']}"

        self.wait_screen_message = parameters['wait_screen_message']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']

        self.stim_number = parameters['stim_number']
        self.stim_length = parameters['stim_length']
        self.time_cross = parameters['time_cross']
        self.time_target = parameters['time_target']
        self.time_flash = parameters['time_flash']
        self.timing = [self.time_target,
                       self.time_cross,
                       self.time_flash]

        self.color = [parameters['target_letter_color'],
                      parameters['fixation_color'],
                      parameters['stim_color']]

        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['stim_height']

        self.is_txt_stim = parameters['is_txt_stim']
        self.eeg_buffer = parameters['eeg_buffer_len']
        self.copy_phrase = parameters['task_text']
        self.spelled_letters_count = int(
            parameters['spelled_letters_count'])
        if self.spelled_letters_count > len(self.copy_phrase):
            self.logger.debug('Already spelled letters exceeds phrase length.')
            self.spelled_letters_count = 0

        self.max_seq_length = parameters['max_seq_len']
        self.max_seconds = parameters['max_minutes'] * 60  # convert to seconds
        self.fake = fake
        self.language_model = language_model
        self.signal_model = signal_model
        self.down_sample_rate = parameters['down_sampling_rate']

        self.filter_low = self.parameters['filter_low_pass']
        self.filter_high = self.parameters['filter_high_pass']
        self.fitler_order = self.parameters['filter_order']
        self.notch_filter_frequency = self.parameters['notch_filter_frequency']

        self.min_num_seq = parameters['min_seq_len']
        self.collection_window_len = parameters['collection_window_after_trial_length']

        self.static_offset = parameters['static_trigger_offset']
        self.show_feedback = parameters['show_feedback']
        self.feedback_color = parameters['feedback_message_color']

        if self.show_feedback:
            self.feedback = VisualFeedback(
                self.window, self.parameters, self.experiment_clock)

    def execute(self):
        self.logger.debug('Starting Copy Phrase Task!')

        # already correctly spelled letters
        text_task = str(self.copy_phrase[0:self.spelled_letters_count])
        task_list = [(str(self.copy_phrase),
                      str(self.copy_phrase[0:self.spelled_letters_count]))]

        # Try Initializing Copy Phrase Wrapper:
        copy_phrase_task = CopyPhraseWrapper(
            self.min_num_seq,
            self.max_seq_length,
            signal_model=self.signal_model,
            fs=self.daq.device_info.fs,
            k=2,
            alp=self.alp,
            task_list=task_list,
            lmodel=self.language_model,
            is_txt_stim=self.is_txt_stim,
            device_name=self.daq.device_info.name,
            device_channels=self.daq.device_info.channels,
            stimuli_timing=[self.time_cross, self.time_flash],
            backspace_prob=self.parameters['lm_backspace_prob'],
            backspace_always_shown=self.parameters['backspace_always_shown'],
            filter_high=self.filter_high,
            filter_low=self.filter_low,
            filter_order=self.fitler_order,
            notch_filter_frequency=self.notch_filter_frequency)

        # Set new epoch (whether to present a new epoch),
        #   run (whether to cont. session),
        #   sequence counter (how many seq have occured).
        #   epoch counter and index
        #   (what epoch, and how many sequences within it)
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

            if self.copy_phrase[0:len(text_task)] == text_task:
                # if correctly spelled so far, get the next unspelled letter.
                target_letter = self.copy_phrase[len(text_task)]
            else:
                # otherwise target is the backspace char.
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
            self.rsvp.stimuli_sequence = ele_sti[0]
            if self.is_txt_stim:
                self.rsvp.stimuli_colors = color_sti[0]
            self.rsvp.stimuli_timing = timing_sti[0]

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

            # Delete calibration
            if seq_counter == 0:
                del sequence_timing[0]

            # reshape the data and triggers as needed for later modules
            raw_data, triggers, target_info = \
                process_data_for_decision(
                    sequence_timing,
                    self.daq,
                    self.window,
                    self.parameters,
                    self.rsvp.first_stim_time,
                    self.static_offset)

            # Uncomment this to turn off fake decisions, but use fake data.
            # self.fake = False
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

                # Evaluate this sequence
                (target_letter, text_task, run) = \
                    fake_copy_phrase_decision(self.copy_phrase,
                                              target_letter,
                                              text_task)

                # here we assume, in fake mode, all sequences result in a selection.
                last_selection = text_task[-1]
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
                                                       target_info, self.collection_window_len)

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
                        .evidence_history['ERP'][-1]
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
                last_selection = copy_phrase_task.decision_maker.last_selection

            # if a letter was selected and feedback enabled, show the chosen letter
            if new_epoch and self.show_feedback:
                self.feedback.administer(
                    last_selection,
                    message='Selected:',
                    line_color=self.feedback_color,
                    fill_color=self.feedback_color)

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
                    self.logger.debug('COPYPHRASE ERROR: Max tries exceeded. To allow for more tries '
                                      'adjust the max_seq_len parameter.')
                if max_time_exceeded:
                    self.logger.debug('COPYPHRASE ERROR: Max time exceeded. To allow for more time '
                                      'adjust the max_minutes parameter.')
                run = False

            # Increment sequence counter
            seq_counter += 1

        # Update task state and reset the static
        self.rsvp.update_task_state(text=text_task, color_list=['white'])

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
        return self.TASK_NAME


def _init_copy_phrase_display(
        parameters, win, daq, static_clock, experiment_clock):
    return CopyPhraseDisplay(
        win,
        static_clock,
        experiment_clock,
        daq.marker_writer,
        info_text=parameters['info_text'],
        static_task_text=parameters['task_text'],
        task_text='****',
        info_color=parameters['info_color'],
        info_pos=(parameters['text_pos_x'],
                  parameters['text_pos_y']),
        info_height=parameters['info_height'],
        info_font=parameters['info_font'],
        task_color=[parameters['task_color']],
        task_font=parameters['task_font'],
        task_height=parameters['task_height'],
        stim_font=parameters['stim_font'],
        stim_pos=(parameters['stim_pos_x'],
                 parameters['stim_pos_y']),
        stim_height=parameters['stim_height'],
        stim_sequence=['a'] * 10,
        stim_colors=[parameters['stim_color']] * 10,
        stim_timing=[3] * 10,
        is_txt_stim=parameters['is_txt_stim'],
        trigger_type=parameters['trigger_type'],
        space_char=parameters['stim_space_char'])
