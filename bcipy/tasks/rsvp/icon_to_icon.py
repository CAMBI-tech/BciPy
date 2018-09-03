from psychopy import core
from itertools import repeat
import csv
import datetime
from os.path import dirname, basename

from bcipy.tasks.task import Task

from bcipy.display.rsvp.rsvp_disp_modes import IconToIconDisplay

from bcipy.helpers.stimuli_generation import generate_icon_match_images

from bcipy.helpers.triggers import _write_triggers_from_sequence_calibration

from bcipy.helpers.signal_model_related import CopyPhraseWrapper

from bcipy.helpers.save import _save_session_related_data

from bcipy.feedback.visual.visual_feedback import VisualFeedback

from bcipy.helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, process_data_for_decision,
    trial_complete_message, get_user_input)

import glob
import logging
from os import path

from psychopy import logging as lg
lg.console.setLevel(logging.WARNING)

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
        classifier : loaded pickle file,
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

    def __init__(
            self, win, daq, parameters, file_save, classifier, lmodel, fake, is_word, auc_filename):

        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = parameters['task_buffer_len']
        self.alp = alphabet(parameters)
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
        self.lmodel = lmodel
        self.classifier = classifier
        self.auc_filename = auc_filename

        self.image_path = parameters['path_to_presentation_images']
        self.task_height = parameters['height_task']

        self.is_txt_sti = False

        self.min_num_seq = parameters['min_seq_len']
        self.word_matching_text_size = parameters['word_matching_text_size']
        self.collection_window_len = parameters['collection_window_after_trial_length']
        self.data_save_path = parameters['data_save_loc']


    def execute(self):
        image_array, timing_array = generate_icon_match_images(self.len_sti,
                                                           self.image_path,
                                                           self.num_sti,
                                                           self.timing,
                                                           self.is_word)

        #Get all png images in image path
        alp_image_array = glob.glob(self.image_path + '*.png')

        #Remove plus image from array
        for image in alp_image_array:
            if image.endswith('PLUS.png'):
                alp_image_array.remove(image)

        if self.is_word:
            image_name_array = glob.glob(self.image_path + '*.png')
            for image in image_name_array:
                image_name_array[image_name_array.index(image)] = path.basename(image)
            alp_image_array.extend(image_name_array)

        for image in alp_image_array:
            alp_image_array[alp_image_array.index(image)] = image.split('/')[-1].split('.')[0]

        self.alp = alp_image_array

        # Try Initializing Copy Phrase Wrapper:
        #       (sig_pro, decision maker, signal_model)
        try:
            copy_phrase_task = CopyPhraseWrapper(self.min_num_seq, self.max_seq_length, signal_model=self.classifier, fs=self.daq.device_info.fs,
                                                 k=2, alp=self.alp, task_list=['unnecessary_string', 'unnecessary_string'],
                                                 lmodel=self.lmodel,
                                                 is_txt_sti=self.is_txt_sti,
                                                 device_name=self.daq.device_info.name,
                                                 device_channels=self.daq.device_info.channels)
        except Exception as e:
            print("Error initializing Copy Phrase Task")
            raise e

        run = True
        new_epoch = True
        epoch_index = 0
        correct_trials = 0

        # Init session data and save before beginning
        data = {
            'session': self.file_save,
            'session_type': 'Icon to Icon Matching',
            'paradigm': 'RSVP',
            'epochs': {},
            'total_time_spent': self.experiment_clock.getTime(),
            'total_number_epochs': 0,
        }

        # Save session data
        _save_session_related_data(self.session_save_location, data)

        # Check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        current_trial = 0
        while run:
            # check user input to make sure we should be going
            if not get_user_input(self.rsvp, self.wait_screen_message,
                                  self.wait_screen_message_color):
                break

            if new_epoch:
                # Init an epoch, getting initial stimuli
                new_epoch, sti = copy_phrase_task.initialize_epoch()

                #If correct decisions are being faked, make sure that we always
                #are starting a new epoch
                if self.fake:
                    new_epoch = True

                # Increase epoch number and reset epoch index
                data['epochs'][current_trial] = {}
                epoch_index = 0
            else:
                epoch_index += 1

            data['epochs'][current_trial][epoch_index] = {}

            if current_trial < len(image_array) or not new_epoch:
                self.rsvp.sti.height = self.stimuli_height

                self.rsvp.stim_sequence = image_array[current_trial]
                self.rsvp.time_list_sti = timing_array
                #Change size of target word if we are in word matching mode
                if self.is_word:
                    #Generate list whose length is the length of the stimuli sequence, filled with the stimuli height
                    self.rsvp.size_list_sti = list(repeat(self.stimuli_height, len(self.rsvp.stim_sequence) + 1))
                    #Set the target word font size to the font size defined in parameters
                    self.rsvp.size_list_sti[0] = self.word_matching_text_size

                core.wait(self.buffer_val)

                self.rsvp.update_task_state(self.rsvp.stim_sequence[0], self.task_height, 'yellow', self.rsvp.win.size, self.is_word)

                # Do the sequence
                sequence_timing = self.rsvp.do_sequence()

                self.first_stim_time = self.rsvp.first_stim_time

                # Write triggers to file
                _write_triggers_from_sequence_calibration(
                    sequence_timing,
                    self.trigger_file)

                # Wait for a time
                core.wait(self.buffer_val)

                # reshape the data and triggers as needed for later modules
                raw_data, triggers, target_info = \
                    process_data_for_decision(sequence_timing, self.daq, self.window,
                        self.parameters, self.first_stim_time)

                #self.fake = False

                display_stimulus = self.rsvp.stim_sequence[0]

                display_message = False
                if self.fake:
                    # Construct Data Record
                    data['epochs'][current_trial][epoch_index] = {
                        'stimuli': image_array[current_trial],
                        'eeg_len': len(raw_data),
                        'timing_sti': timing_array,
                        'triggers': triggers,
                        'target_info': target_info,
                        'target_letter': display_stimulus
                        }
                    correct_decision = True
                    display_message = True
                    message_color = 'green'
                    current_trial += 1
                    correct_trials += 1
                    new_epoch = True
                    if self.is_word:
                        display_stimulus = self.image_path + self.rsvp.stim_sequence[0] + '.png'
                else:
                    new_epoch, sti = \
                        copy_phrase_task.evaluate_sequence(raw_data, triggers,
                                                           target_info, self.collection_window_len)

                    # Construct Data Record
                    data['epochs'][current_trial][epoch_index] = {
                        'stimuli': image_array[current_trial],
                        'eeg_len': len(raw_data),
                        'timing_sti': timing_array,
                        'triggers': triggers,
                        'target_info': target_info,
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

                    #Test whether to display feedback message, and what color
                    #the message should be
                    if new_epoch:
                        if self.is_word:
                            decide_image_path = self.image_path + copy_phrase_task.decision_maker.last_selection + '.png'
                        else:
                            decide_image_path = copy_phrase_task.decision_maker.last_selection + '.png'
                        correct_decision = decide_image_path == self.rsvp.stim_sequence[0]
                        display_stimulus = decide_image_path
                        current_trial += 1
                        if correct_decision:
                            message_color = 'green'
                            correct_trials += 1
                        else:
                            message_color = 'red'
                        display_message = True
                    else:
                        display_message = False

                if display_message:
                    #Display feedback about whether decision was correct
                    visual_feedback = VisualFeedback(
                    display=self.window, parameters=self.parameters, clock=self.experiment_clock)
                    stimulus = display_stimulus
                    visual_feedback.message_color = message_color
                    visual_feedback.administer(stimulus, compare_assertion=None, message='Decision:')


                # Update time spent and save data
                data['total_time_spent'] = self.experiment_clock.getTime()
                data['total_number_epochs'] = current_trial
                _save_session_related_data(self.session_save_location, data)

                # Decide whether to keep the task going
                max_tries_exceeded = current_trial >= self.max_seq_length
                max_time_exceeded = data['total_time_spent'] >= self.max_seconds
                if (max_tries_exceeded or max_time_exceeded):
                    if max_tries_exceeded:
                        logging.debug("Max tries exceeded: to allow for more tries"
                                      " adjust the Maximum Sequence Length "
                                      "(max_seq_len) parameter.")
                    if max_time_exceeded:
                        logging.debug("Max time exceeded. To allow for more time "
                                      "adjust the max_minutes parameter.")
                    run = False

            else:
                run = False

        #Write trial data to icon_data.csv in file save location
        with open(f"{self.file_save}/icon_data.csv", 'w+') as icon_output_csv:
            icon_output_writer = csv.writer(icon_output_csv, delimiter=',')
            icon_output_writer.writerow(['Participant ID', dirname(self.file_save).replace(self.data_save_path, '')])
            icon_output_writer.writerow(['Date/Time', datetime.datetime.now()])
            if self.auc_filename:
                icon_output_writer.writerow(['Calibration AUC', basename(self.auc_filename).replace('.pkl', '')])
            temp_epoch_index = 1 if epoch_index == 0 else epoch_index
            temp_current_trial = 1 if current_trial == 0 else current_trial
            icon_output_writer.writerow(['Percentage of correctly selected icons', (correct_trials / (temp_current_trial * temp_epoch_index)) * 100])
            icon_output_writer.writerow(['Task type', ('Icon to word' if self.is_word else 'Icon to icon')])

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

        return self.file_save


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
