"""RSVP Icon to Icon Matching Task."""
import csv
import datetime
from os.path import basename, dirname
from typing import List, Tuple

from psychopy import core

from bcipy.display.rsvp.mode.icon_to_icon import IconToIconDisplay
from bcipy.feedback.visual.visual_feedback import FeedbackType, VisualFeedback
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.signal_model import CopyPhraseWrapper
from bcipy.helpers.task import (alphabet, generate_targets, get_user_input,
                                process_data_for_decision,
                                trial_complete_message)
from bcipy.helpers.triggers import write_triggers_from_sequence_icon_to_icon
from bcipy.language_model.random_language_model import RandomLm
from bcipy.tasks.task import Task


class RSVPIconToIconTask(Task):
    """RSVP Icon to Icon Matching Task.

    Parameters
    ----------
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        daq : object,
            data acquisition object initialized for the desired protocol
        file_save : str,
            path location of where to save data from the session
        signal_model : loaded pickle file,
            trained signal_model, loaded before session started
        fake : boolean, optional,
            boolean to indicate whether this is a fake session or not.
        is_word: boolean,
            boolean to indicate whether or not this is an icon to word matching session
        auc_filename: str,
            name of the loaded pickle file
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """
    TASK_NAME = 'RSVP Icon to Icon Task'

    def __init__(self, win, daq, parameters, file_save, signal_model,
                 language_model, fake, is_word, auc_filename):
        super(RSVPIconToIconTask, self).__init__()
        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = parameters['task_buffer_len']

        self.image_path = parameters['path_to_presentation_images']
        # Alphabet is comprised of the image base names
        self.alp = alphabet(parameters, include_path=False)

        self.rsvp = _init_icon_to_icon_display_task(self.parameters,
                                                    self.window, self.daq,
                                                    self.static_clock,
                                                    self.experiment_clock,
                                                    is_word)
        self.file_save = file_save
        self.is_word = is_word

        trigger_save_location = f'{self.file_save}/{parameters["trigger_file_name"]}'
        self.trigger_file = open(trigger_save_location, 'w+')
        self.session_save_location = f'{self.file_save}/{parameters["session_file_name"]}'

        self.wait_screen_message = parameters['wait_screen_message']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']

        self.stim_number = parameters['stim_number']
        self.stim_length = parameters['stim_length']
        self.timing = [
            parameters['time_target'], parameters['time_cross'],
            parameters['time_flash']
        ]

        self.color = [
            parameters['target_letter_color'], parameters['fixation_color'],
            parameters['stim_color']
        ]

        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['stim_height']

        self.eeg_buffer = parameters['eeg_buffer_len']

        self.max_seconds = parameters['max_minutes'] * 60  # convert to seconds
        self.max_seq_length = parameters['max_seq_len']
        self.max_seq_per_trial = parameters['max_seq_per_trial']
        self.fake = fake

        self.language_model = language_model or RandomLm(alphabet=self.alp)
        self.signal_model = signal_model
        self.auc_filename = auc_filename

        self.task_height = parameters['task_height']

        self.is_txt_stim = False

        self.min_num_seq = parameters['min_seq_len']
        self.word_matching_text_size = parameters['word_matching_text_size']
        self.collection_window_len = parameters[
            'collection_window_after_trial_length']

        self.data_save_path = parameters['data_save_loc']

        match_type = 'Word' if self.is_word else 'Icon'
        self.session_description = f'Icon to {match_type} Matching'

    def img_path(self, alphabet_item):
        """Return the full image path for the given alphabet item. If the item
        ends with .png it's returned as is."""
        if alphabet_item.endswith('.png'):
            return alphabet_item
        return self.image_path + alphabet_item + '.png'

    def init_session_data(self, save: bool = True):
        """Initializes the session data; saved to session.json file.
        Parameters:
        -----------
            save - whether to save the data to disk.
        Returns:
        --------
            data - map with initial session data.
        """
        data = {
            'session': self.file_save,
            'session_type': self.session_description,
            'paradigm': 'RSVP',
            'epochs': {},
            'total_time_spent': self.experiment_clock.getTime(),
            'total_number_epochs': 0,
        }

        if save:
            _save_session_related_data(self.session_save_location, data)
        return data

    def await_start(self) -> bool:
        """Wait for user input to either exit or start"""
        self.logger.debug('Awaiting user start.')
        should_continue = get_user_input(self.rsvp,
                                         self.wait_screen_message,
                                         self.wait_screen_message_color,
                                         first_run=True)
        return should_continue

    def user_wants_to_continue(self) -> bool:
        """Check if user wants to continue or terminate.
        Returns True to continue."""
        should_continue = get_user_input(self.rsvp,
                                         self.wait_screen_message,
                                         self.wait_screen_message_color,
                                         first_run=False)
        if not should_continue:
            self.logger.debug('User wants to exit.')
        return should_continue

    def present_sequence(self,
                         seq: List[str],
                         durations: List[float],
                         show_target: bool = False):
        """Present the given sequence and return the trigger timing info.
        Parameters:
        ----------
          seq - list of alphabet items to present
          duration - list of durations (float) to present each item
          show_target - optional item to highlight the first sequence item as
            the target (displayed in the header and outlined).
        Returns:
        --------
            sequence timings - list of tuples representing the letter and time
                that it was presented.
        """
        # Sequences passed to rsvp to display should be a list of image paths
        # except for a word target.
        word_target = show_target and self.is_word
        self.rsvp.stimuli_sequence = [
            item if i == 0 and word_target else self.img_path(item)
            for i, item in enumerate(seq)
        ]
        self.rsvp.stimuli_timing = durations

        if self.is_word:
            if show_target:
                # Change size of target word if we are in word matching mode
                # and the word is presented.
                word_height = self.word_matching_text_size
                self.rsvp.size_list_sti = [
                    word_height if i == 0 else self.stimuli_height
                    for i, _ in enumerate(seq)
                ]
            else:
                self.rsvp.size_list_sti = []

        core.wait(self.buffer_val)

        # Present the Target and place it in the header
        if show_target:
            self.rsvp.update_task_state(self.rsvp.stimuli_sequence[0],
                                        self.task_height, 'yellow',
                                        self.rsvp.window.size, self.is_word)

        self.rsvp.highlight_first_stim = show_target
        # Note that returned triggers use image basenames only.
        sequence_timing = self.rsvp.do_sequence()
        return sequence_timing

    def display_feedback(self, selection: str, correct: bool):
        """Display feedback for the given selection."""
        feedback = VisualFeedback(display=self.window,
                                  parameters=self.parameters,
                                  clock=self.experiment_clock)
        feedback.message_color = 'green' if correct else 'red'
        feedback.administer(self.img_path(selection),
                            compare_assertion=None,
                            message='Decision: ',
                            stimuli_type=FeedbackType.IMAGE)

    def write_data(self, correct_trials: int, selections: int):
        """Write trial data to icon_data.csv in file save location."""

        with open(f"{self.file_save}/icon_data.csv", 'w+') as icon_output_csv:
            writer = csv.writer(icon_output_csv, delimiter=',')
            writer.writerow([
                'Participant ID',
                dirname(self.file_save).replace(self.data_save_path, '')
            ])
            writer.writerow(['Date/Time', datetime.datetime.now()])
            if self.auc_filename:
                writer.writerow([
                    'Calibration AUC',
                    basename(self.auc_filename).replace('.pkl', '')
                ])

            current_trial = 1 if selections == 0 else selections

            writer.writerow([
                'Percentage of correctly selected icons',
                (correct_trials / current_trial) * 100
            ])
            writer.writerow(['Task type', self.session_description])

    def exit_display(self):
        """Close the UI and cleanup"""
        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self.buffer_val)

        if self.daq.is_calibrated:
            write_triggers_from_sequence_icon_to_icon(
                sequence_timing=['offset', self.daq.offset],
                trigger_file=self.trigger_file,
                target=None,
                target_displayed=False,
                offset=True)
        # Close this sessions trigger file and return some data
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

    def init_copy_phrase_task(self, task_list: List[Tuple]):
        """Initialize the CopyPhraseWrapper
        Parameters:
        -----------
            task_list: List of (sequence to match, items matched so far).
        Returns:
        --------
            initialized CopyPhraseWrapper
        """
        return CopyPhraseWrapper(
            self.min_num_seq,
            self.max_seq_per_trial,
            signal_model=self.signal_model,
            fs=self.daq.device_info.fs,
            k=2,
            alp=self.alp,
            task_list=task_list,
            lmodel=self.language_model,
            is_txt_stim=self.is_txt_stim,
            device_name=self.daq.device_info.name,
            device_channels=self.daq.device_info.channels,
            decision_threshold=self.parameters['decision_threshold'],
            stimuli_timing=self.timing[1:])  # time_cross and time_flash

    def stoppage_criteria_ok(self, total_sequences, total_time) -> bool:
        """Returns True if experiment is currently within params, False if
        total sequences or total time exceeds configured values."""
        if total_sequences >= self.max_seq_length:
            self.logger.debug('Max tries exceeded: to allow for more tries'
                              ' adjust the Maximum Sequence Length '
                              '(max_seq_len) parameter.')
            return False

        if total_time >= self.max_seconds:
            self.logger.debug('Max time exceeded. To allow for more time '
                              'adjust the max_minutes parameter.')
            return False
        return True

    def update_session_data(self, data, epoch, save=True):
        """Update the session data and optionally save."""
        data['total_time_spent'] = self.experiment_clock.getTime()
        data['total_number_epochs'] = epoch.epoch_counter
        data['epochs'] = epoch.data
        if save:
            _save_session_related_data(self.session_save_location, data)

    def execute(self):
        """Execute the task"""
        self.logger.debug('Starting Icon to Icon Task!')

        icons = generate_targets(self.alp, self.stim_number)
        self.logger.debug('Icon sequence: %s', icons)

        selections = []
        copy_phrase_task = self.init_copy_phrase_task(task_list=[(icons, [])])
        epoch = EpochManager(icons, copy_phrase_task, self.timing[0])
        correct_trials = 0

        data = self.init_session_data(save=True)
        run = self.await_start()

        epoch.next()
        while run and self.user_wants_to_continue():

            sequence_timing = self.present_sequence(
                epoch.current_sequence,
                epoch.current_durations,
                show_target=epoch.first_sequence)

            # Write triggers to file
            write_triggers_from_sequence_icon_to_icon(
                sequence_timing,
                self.trigger_file,
                epoch.target,
                target_displayed=epoch.first_sequence)

            core.wait(self.buffer_val)

            # Delete calibration
            if epoch.first_stimulus:
                del sequence_timing[0]

            # Delete the target presentation
            if epoch.first_sequence:
                del sequence_timing[0]

            # Reshape the data and triggers as needed for analysis.
            raw_data, triggers, target_info = process_data_for_decision(
                sequence_timing, self.daq, self.window, self.parameters,
                self.rsvp.first_stim_time)

            # Data Record for session.json
            entry = {
                'stimuli': epoch.stimuli,
                'eeg_len': len(raw_data),
                'timing_sti': epoch.timing_sti,
                'triggers': triggers,
                'target_info': target_info,
                'target_letter': epoch.target
            }

            if self.fake:
                new_epoch = True
                correct_trials += 1
                selections.append(epoch.target)
                self.display_feedback(selection=epoch.target, correct=True)
            else:
                # Evaluate the data and make a decision.
                decision_made, new_sti = copy_phrase_task.evaluate_sequence(
                    raw_data, triggers, target_info,
                    self.collection_window_len)
                new_epoch = decision_made

                # Add the evidence to the data record.
                ev_hist = copy_phrase_task.conjugator.evidence_history
                likelihood = copy_phrase_task.conjugator.likelihood
                entry['lm_evidence'] = ev_hist['LM'][0].tolist()
                entry['eeg_evidence'] = ev_hist['ERP'][-1].tolist()
                entry['likelihood'] = likelihood.tolist()

                if decision_made:
                    selection = copy_phrase_task.decision_maker.last_selection
                    selections.append(selection)
                    correct = selection == epoch.target
                    if correct:
                        correct_trials += 1
                    self.display_feedback(selection, correct)

            epoch.add_stim_data(entry)
            self.update_session_data(data, epoch, save=True)
            if new_epoch:
                epoch.next()
            else:
                epoch.next_stimulus(new_sti)

            run = self.stoppage_criteria_ok(epoch.seq_counter,
                                            data['total_time_spent'])
            # end while loop

        self.write_data(correct_trials, len(selections))
        self.exit_display()
        return self.file_save

    @classmethod
    def label(cls):
        return RSVPIconToIconTask.TASK_NAME

    def name(self):
        return RSVPIconToIconTask.TASK_NAME


def _init_icon_to_icon_display_task(parameters, win, daq, static_clock,
                                    experiment_clock, is_word):
    return IconToIconDisplay(win,
                             static_clock,
                             experiment_clock,
                             daq.marker_writer,
                             info_text=parameters['info_text'],
                             info_color=parameters['info_color'],
                             info_pos=(parameters['text_pos_x'],
                                       parameters['text_pos_y']),
                             info_height=parameters['info_height'],
                             info_font=parameters['info_font'],
                             task_color=['black'],
                             task_font=parameters['task_font'],
                             task_height=parameters['task_height'],
                             stim_font=parameters['stim_font'],
                             stim_pos=(parameters['stim_pos_x'],
                                       parameters['stim_pos_y']),
                             stim_height=parameters['stim_height'],
                             stim_sequence=['a'] * 10,
                             stim_colors=[parameters['stim_color']] * 10,
                             stim_timing=[3] * 10,
                             is_txt_stim=False,
                             trigger_type=parameters['trigger_type'],
                             is_word=is_word)


class EpochManager():
    """Manages the state required for tracking epochs.
    Parameters:
    -----------
        icons - sequence of icons to match
        copy_phrase_task - CopyPhraseWrapper
        time_target - time to present a target stimulus.
    """

    def __init__(self, icons, copy_phrase_task, time_target):
        self.copy_phrase_task = copy_phrase_task
        self.target = None
        self.time_target = time_target
        self.icons = icons

        self.current_sequence = []
        self.current_durations = []

        self.epoch_counter = -1
        self.epoch_index = 0
        self.data = {}
        self.seq_counter = 0

    #  Public API
    def next(self):
        """Initialize the next epoch/target"""
        self._increment_epoch()
        self._new_epoch_stimulus()
        self.seq_counter += 1

    def next_stimulus(self, sti):
        """Add another sequence for the current target.
        Parameters:
        -----------
            sti - tuple of (List[alphabet], List[durations], List[colors])
        """
        self.epoch_index += 1
        self.seq_counter += 1
        self._set_stimulus(sti)

    @property
    def first_sequence(self) -> bool:
        """Is the current sequence the first one displayed for this epoch?"""
        return self.epoch_index == 0

    @property
    def first_stimulus(self) -> bool:
        """Is this the first stimulus presented?"""
        return self.epoch_index == 0 and self.first_sequence

    @property
    def stimuli(self) -> List[str]:
        """Stimuli to write to the session.json; excludes the target
        presentation"""
        if self.first_sequence:
            return self.current_sequence[1:]
        else:
            return self.current_sequence

    @property
    def timing_sti(self) -> List[float]:
        """Timing Stimuli to write to the session.json; excludes the target
        presentation"""
        if self.first_sequence:
            return self.current_durations[1:]
        else:
            return self.current_durations

    def add_stim_data(self, data):
        """Add stimulus data for the current epoch_counter/index"""
        self.data[self.epoch_counter][self.epoch_index] = data

    # Internally used methods
    def _increment_epoch(self):
        """Increment the counters and select a new target."""
        self.epoch_counter += 1
        self.epoch_index = 0
        self.target = self.icons[self.epoch_counter]
        self.data[self.epoch_counter] = {}

    def _new_epoch_stimulus(self):
        """Init an epoch, generate initial stimuli seq."""
        _, sti = self.copy_phrase_task.initialize_epoch()
        self._set_stimulus(sti)
        self._new_sequence_mods()

    def _set_stimulus(self, sti: Tuple):
        """Sets the current sequence and current durations from the sti output
        from copy_phrase_task.
        Parameters:
        -----------
            sti - tuple of (List[alphabet], List[durations], List[colors])
        """
        self.current_sequence = sti[0][0]
        self.current_durations = sti[1][0]

    def _new_sequence_mods(self):
        """Make any modifications to the current sequence"""
        # Insert target for initial display
        self.current_sequence.insert(0, self.target)
        self.current_durations.insert(0, self.time_target)
