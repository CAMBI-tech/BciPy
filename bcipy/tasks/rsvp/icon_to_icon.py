import csv
import datetime
import glob
import logging
import random
from itertools import repeat
from os import path
from os.path import basename, dirname
from typing import Dict, List, Sequence, Tuple

from psychopy import core

from bcipy.display.rsvp.rsvp_disp_modes import IconToIconDisplay
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.bci_task_related import (alphabet,
                                            fake_copy_phrase_decision,
                                            get_user_input,
                                            process_data_for_decision,
                                            trial_complete_message)
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.signal_model_related import CopyPhraseWrapper
from bcipy.helpers.stimuli_generation import generate_icon_match_images
from bcipy.helpers.triggers import write_triggers_from_sequence_icon_to_icon
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

    def __init__(
            self, win, daq, parameters, file_save, signal_model, language_model, fake, is_word, auc_filename):
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
        self.alp = [
            path.splitext(path.basename(img))[0]
            for img in glob.glob(self.image_path + "*.png")
            if not img.endswith("PLUS.png")
        ]

        self.rsvp = _init_icon_to_icon_display_task(
            self.parameters, self.window, self.daq,
            self.static_clock, self.experiment_clock, is_word)
        self.file_save = file_save
        self.is_word = is_word

        trigger_save_location = f"{self.file_save}/{parameters['triggers_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w+')
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

        self.eeg_buffer = parameters['eeg_buffer_len']

        self.max_seconds = parameters['max_minutes'] * 60  # convert to seconds
        self.max_seq_length = parameters['max_seq_len']
        self.fake = fake
        self.language_model = language_model
        self.signal_model = signal_model
        self.auc_filename = auc_filename

        self.task_height = parameters['height_task']

        self.is_txt_sti = False

        self.min_num_seq = parameters['min_seq_len']
        self.word_matching_text_size = parameters['word_matching_text_size']
        self.collection_window_len = parameters['collection_window_after_trial_length']
        self.data_save_path = parameters['data_save_loc']

        match_type = "Word" if self.is_word else "Icon"
        self.session_description = f'Icon to {match_type} Matching'

    def img_path(self, alphabet_item):
        """Return the full image path for the given alphabet item."""
        if alphabet_item.startswith(self.image_path):
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
        logging.debug("Awaiting user start.")
        should_continue = get_user_input(
            self.rsvp,
            self.wait_screen_message,
            self.wait_screen_message_color,
            first_run=True)
        return should_continue

    def user_wants_to_continue(self) -> bool:
        """Check if user wants to continue or terminate. 
        Returns True to continue."""
        should_continue = get_user_input(
            self.rsvp,
            self.wait_screen_message,
            self.wait_screen_message_color,
            first_run=False)
        if not should_continue:
            logging.debug("User wants to exit.")
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
        # TODO: is this necessary?
        self.rsvp.sti.height = self.stimuli_height

        # Sequences passed to rsvp to display should be a list of image paths.
        self.rsvp.stim_sequence = [self.img_path(item) for item in seq]
        self.rsvp.time_list_sti = durations

        # TODO: test this code
        # Change size of target word if we are in word matching mode
        if self.is_word:
            # Generate list whose length is the length of the stimuli sequence, filled with the stimuli height
            self.rsvp.size_list_sti = list(
                repeat(self.stimuli_height,
                       len(self.rsvp.stim_sequence) + 1))
            # Set the target word font size to the font size defined in parameters
            self.rsvp.size_list_sti[0] = self.word_matching_text_size

        core.wait(self.buffer_val)

        # Present the Target and place it in the header
        if show_target:
            self.rsvp.update_task_state(self.rsvp.stim_sequence[0],
                                        self.task_height, 'yellow',
                                        self.rsvp.win.size, self.is_word)

        # Note that returned triggers use image basenames only.
        sequence_timing = self.rsvp.do_sequence()
        return sequence_timing

    def display_feedback(self, selection : str, correct: bool):
        """Display feedback for the given selection."""
        feedback = VisualFeedback(
            display=self.window,
            parameters=self.parameters,
            clock=self.experiment_clock)
        feedback.message_color = 'green' if correct else 'red'
        feedback.administer(
            self.image_path(selection),
            compare_assertion=None,
            message='Decision: ')

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
            self.max_seq_length,
            signal_model=self.signal_model,
            fs=self.daq.device_info.fs,
            k=2,
            alp=self.alp,
            task_list=task_list,
            lmodel=self.language_model,
            is_txt_sti=False, # TODO: is this True for icon to word?
            device_name=self.daq.device_info.name,
            device_channels=self.daq.device_info.channels,
            stimuli_timing=self.timing[1:])  # time_cross and time_flash

    def stoppage_criteria_ok(self, total_sequences, total_time) -> bool:
        """Returns True if experiment is currently within params, False if 
        total sequences or total time exceeds configured values."""
        if total_sequences >= self.max_seq_length:
            logging.debug("Max tries exceeded: to allow for more tries"
                          " adjust the Maximum Sequence Length "
                          "(max_seq_len) parameter.")
            return False

        if total_time >= self.max_seconds:
            logging.debug("Max time exceeded. To allow for more time "
                          "adjust the max_minutes parameter.")
            return False
        return True


    def execute(self):
        self.logger.debug('Starting Icon to Icon Task!')

        icons = [random.choice(self.alp) for _ in range(self.num_sti)]
        selections = []
        logging.debug(f"Icon sequence: {icons}")
        copy_phrase_task = self.init_copy_phrase_task(task_list=[(icons, [])])

        new_epoch = True  # Whether to present a new epoch.
        epoch_counter = -1  # incremented for each new target
        epoch_index = 0  # incremented for each attempt at matching a target.
        seq_counter = 0  # Total number of seqs that have been presented.
        correct_trials = 0

        # Used for stimulus presentation
        current_seq = []
        current_durations = []

        data = self.init_session_data(save=True)
        run = self.await_start()

        while run and self.user_wants_to_continue():

            if new_epoch:
                # Initialize epoch
                epoch_counter += 1
                epoch_index = 0
                target = icons[epoch_counter]

                # Init an epoch, generate initial stimuli seq.
                _, sti = copy_phrase_task.initialize_epoch()
                current_seq = sti[0][0]
                current_durations = sti[1][0]

                # Insert target for initial display
                current_seq.insert(0, target)
                current_durations.insert(0, self.timing[0])
                data['epochs'][epoch_counter] = {}
            else:
                epoch_index += 1

            sequence_timing = self.present_sequence(
                current_seq, current_durations, show_target=epoch_index == 0)

            # Write triggers to file
            write_triggers_from_sequence_icon_to_icon(
                sequence_timing, self.trigger_file, icons, selections)
            
            core.wait(self.buffer_val)

            # Delete calibration
            if seq_counter == 0:
                del sequence_timing[0]

            # Reshape the data and triggers as needed for analysis.
            raw_data, triggers, target_info = process_data_for_decision(
                sequence_timing, self.daq, self.window, self.parameters,
                self.rsvp.first_stim_time)
            # Data Record for session.json
            entry = {
                'stimuli': current_seq,
                'eeg_len': len(raw_data),
                'timing_sti': current_durations,
                'triggers': triggers,
                'target_info': target_info,
                'target_letter': target
            }

            if self.fake:
                new_epoch = True
                correct_trials += 1
                selections.append(target)
                self.display_feedback(selection=target, correct=True)
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
                entry['eeg_evidence'] = ev_hist['ERP'][0].tolist()
                entry['likelihood'] = likelihood.tolist()

                if decision_made:
                    selection = copy_phrase_task.decision_maker.last_selection
                    selections.append(selection)
                    correct = selection == target
                    if correct:
                        correct_trials += 1
                    self.display_feedback(selection, correct)                    
                else:
                    # Update to use a new sequence prepared by the
                    # DecisionMaker based on the evidence.
                    current_seq = new_sti[0][0]
                    current_durations = new_sti[1][0]

            seq_counter += 1
            data['epochs'][epoch_counter][epoch_index] = entry
            data['total_time_spent'] = self.experiment_clock.getTime()
            data['total_number_epochs'] = epoch_counter
            _save_session_related_data(self.session_save_location, data)

            run = self.stoppage_criteria_ok(seq_counter, data['total_time_spent'])
            # end while loop
                
        self.write_data(correct_trials, len(selections))
        self.exit_display()
        return self.file_save

    @classmethod
    def label(cls):
        return RSVPIconToIconTask.TASK_NAME

    def name(self):
        return RSVPIconToIconTask.TASK_NAME


def _init_icon_to_icon_display_task(
        parameters, win, daq, static_clock, experiment_clock, is_word):
    rsvp = IconToIconDisplay(
        window=win, clock=static_clock,
        experiment_clock=experiment_clock,
        marker_writer=daq.marker_writer,
        text_info=parameters['text_text'],
        color_info=parameters['color_text'],
        pos_info=(parameters['pos_text_x'],
                  parameters['pos_text_y']),
        height_info=parameters['txt_height'],
        font_info=parameters['font_text'],
        color_task=['black'],
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
        is_txt_sti=False,
        trigger_type=parameters['trigger_type'],
        is_word=is_word)

    return rsvp
