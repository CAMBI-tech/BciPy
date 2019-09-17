from psychopy import core
from typing import List, Tuple

from bcipy.feedback.visual.level_feedback import LevelFeedback
from bcipy.tasks.task import Task
from bcipy.tasks.exceptions import InsufficientDataException
from bcipy.tasks.rsvp.calibration.calibration import RSVPCalibrationTask
from bcipy.helpers.triggers import _write_triggers_from_sequence_calibration
from bcipy.helpers.stimuli import random_rsvp_calibration_seq_gen, get_task_info
from bcipy.signal.process.filter import bandpass, downsample, notch
from bcipy.helpers.task import (
    trial_complete_message,
    trial_reshaper,
    get_user_input,
    pause_calibration,
    process_data_for_decision)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.signal.process.decomposition.psd import power_spectral_density, PSD_TYPE


class RSVPInterSequenceFeedbackCalibration(Task):
    """RSVP InterSequenceFeedbackCalibration Task uses inter sequence
        feedback to alert user to their current state in order to increase performance
        in a calibration task.

    Calibration task performs an RSVP stimulus sequence to elicit an ERP.
    Parameters will change how many stim and for how long they present.
    Parameters also change color and text / image inputs. This task assumes 5 levels
    of feedback. The parameters for this task are in the feedback_config section.

    Note: The channel and are PSD method used for feedback are hardcoded for piloting.
        In the future, these may be changed to allow easier feedback for different caps.


    Input:
        win (PsychoPy Display Object)
        daq (Data Acquisition Object)
        parameters (Dictionary)
        file_save (String)

    Output:
        file_save (String)
    """
    TASK_NAME = 'RSVP Inter Sequence Feedback Calibration Task'
    # This defines the channel we use to calculate the PSD for feedback. We want to use a
    #   posterior channel. If Oz available, use that!
    PSD_CHANNEL_INDEX = 6

    def __init__(self, win, daq, parameters, file_save):
        super(RSVPInterSequenceFeedbackCalibration, self).__init__()
        self._task = RSVPCalibrationTask(
            win,
            daq,
            parameters,
            file_save)

        self.daq = daq
        self.fs = self.daq.device_info.fs
        self.alp = self._task.alp
        self.rsvp = self._task.rsvp
        self.parameters = parameters
        self.file_save = file_save
        self.enable_breaks = self._task.enable_breaks
        self.window = self._task.window
        self.stim_number = self._task.stim_number
        self.stim_length = self._task.stim_length
        self.is_txt_stim = self.rsvp.is_txt_stim
        self.stimuli_height = self._task.stimuli_height
        self.color = self._task.color
        self.timing = self._task.timing
        self.wait_screen_message = self._task.wait_screen_message
        self.wait_screen_message_color = self._task.wait_screen_message_color

        self.visual_feedback = LevelFeedback(
            display=self.window,
            parameters=self.parameters,
            clock=self._task.experiment_clock)

        self.static_offset = self.parameters['static_trigger_offset']
        self.nonletters = ['+', 'PLUS', 'calibration_trigger']
        self.valid_targets = set(self.alp)

        self.time_flash = self.parameters['time_flash']

        self.downsample_rate = self.parameters['down_sampling_rate']
        self.filtered_sampling_rate = self.fs / self.downsample_rate

        self.device_name = self.daq.device_info.name
        self.channel_map = analysis_channels(
            self.daq.device_info.channels, self.device_name)

        # EDIT ME FOR FEEDBACK CONFIGURATION

        self.feedback_buffer_time = self.parameters['feedback_buffer_time']
        self.feedback_line_color = self.parameters['feedback_line_color']

        self.psd_method = PSD_TYPE.WELCH

        # The channel used to calculate the PSD from RSVP sequence.
        self.psd_channel_index = self.PSD_CHANNEL_INDEX

        # filter parameters
        self.filter_low = self.parameters['filter_low_pass']
        self.filter_high = self.parameters['filter_high_pass']
        self.filter_order = self.parameters['filter_order']
        self.notch_filter_frequency = self.parameters['notch_filter_frequency']

        # get the feedback band of interest
        self.psd_lower_limit = self.parameters['feedback_band_lower_limit']
        self.psd_upper_limit = self.parameters['feedback_band_upper_limit']

        # psd band of interest to use for feeback (low, high)
        self.psd_export_band = (self.psd_lower_limit, self.psd_upper_limit)

        # length of time to use for PSD calculation
        self.trial_length = self.time_flash * self.stim_length

        self.lvl_5_threshold = self.parameters['feedback_level_5_threshold']
        self.lvl_4_threshold = self.parameters['feedback_level_4_threshold']
        self.lvl_3_threshold = self.parameters['feedback_level_3_threshold']
        self.lvl_2_threshold = self.parameters['feedback_level_2_threshold']

        # true/false order is desceding from 5 -> 1 for level
        self.feedback_descending = self.parameters['feedback_level_descending']

    def execute(self):
        self.logger.debug(f'Starting {self.name()}!')
        run = True

        # Check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        # Begin the Experiment
        while run:

            # Get random sequence information given stimuli parameters
            (stimuli_elements, timing_sti,
             color_sti) = random_rsvp_calibration_seq_gen(
                 self.alp,
                 stim_number=self.stim_number,
                 stim_length=self.stim_length,
                 timing=self.timing,
                 is_txt=self.is_txt_stim,
                 color=self.color)

            (task_text, task_color) = get_task_info(self.stim_number,
                                                    self._task.task_info_color)

            # Execute the RSVP sequences
            for sequence_idx in range(len(task_text)):

                # check user input to make sure we should be going
                if not get_user_input(self.rsvp, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

                if self.enable_breaks:
                    pause_calibration(self.window, self.rsvp, sequence_idx,
                                      self.parameters)

                # update task state
                self.rsvp.update_task_state(
                    text=task_text[sequence_idx],
                    color_list=task_color[sequence_idx])

                # Draw and flip screen
                self.rsvp.draw_static()
                self.window.flip()

                # Get height
                self.rsvp.sti.height = self.stimuli_height

                # Schedule a sequence
                self.rsvp.stimuli_sequence = stimuli_elements[sequence_idx]

                # check if text stimuli or not for color information
                if self.is_txt_stim:
                    self.rsvp.stimuli_colors = color_sti[sequence_idx]

                self.rsvp.stimuli_timing = timing_sti[sequence_idx]

                # Wait for a time
                core.wait(self._task.buffer_val)

                # Do the sequence
                last_sequence_timing = self.rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, self._task.trigger_file)

                self.logger.info('[Feedback] Getting Decision')

                position = self._get_feedback_decision(last_sequence_timing)
                self.logger.info(
                    f'[Feedback] Administering feedback position {position}')
                self.visual_feedback.administer(position=position)

                # Wait for a time
                core.wait(self._task.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self._task.buffer_val)

        if self.daq.is_calibrated:
            _write_triggers_from_sequence_calibration(
                ['offset', self.daq.offset], self._task.trigger_file, offset=True)

        # Close this sessions trigger file and return some data
        self._task.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self._task.eeg_buffer)

        return self.file_save

    def _get_feedback_decision(self, sequence_timing):
        # wait some time in order to get enough data from the daq and make the
        #   transition less abrupt to the user
        core.wait(self.feedback_buffer_time)

        # get last stim_length stimuli
        sequence_of_interest = sequence_timing[-self.stim_length:]

        # get data sequence and only use the first 2 stimuli
        data = self._get_data_for_psd(sequence_of_interest[:2])

        # we always want the same data channel in the occipital region and the
        # first of it
        response = power_spectral_density(
            data[self.psd_channel_index][0],
            self.psd_export_band,
            sampling_rate=self.filtered_sampling_rate,
            # plot=True,  # uncomment to see the PSD plot in real time
            method=self.psd_method,
            relative=True)

        self.logger.info(f'[Feedback] Response decision {response}')

        # In the event the calcalated band returns nothing, throw an
        #  error
        if response == 0:
            message = 'PSD calculated for feedback invalid'
            self.logger.error(f'[Feedback] {message}')
            raise InsufficientDataException(message)

        return self._determine_feedback_response(response)

    def _determine_feedback_response(self, response):
        """Determine feedback response.

        Depending on the band chosen to give feedback off, we may need to invert the
            levels. By default, it's in descending order. Set feedback_level_descending
            to false for ascending
        """
        # default condition; for use with SSVEP or PSD that increases with focus to task
        if self.feedback_descending:
            if response > self.lvl_5_threshold:
                return 5
            if response > self.lvl_4_threshold:
                return 4
            if response > self.lvl_3_threshold:
                return 3
            if response > self.lvl_2_threshold:
                return 2
            return 1

        # ascending condition; use with PSD that decrease with focus to task
        else:
            if response < self.lvl_5_threshold:
                return 5
            if response < self.lvl_4_threshold:
                return 4
            if response < self.lvl_3_threshold:
                return 3
            if response < self.lvl_2_threshold:
                return 2
            return 1

    def _get_data_for_psd(self, sequence_timing):
        # get data from the DAQ
        raw_data, triggers, target_info = process_data_for_decision(
            sequence_timing,
            self.daq,
            self.window,
            self.parameters,
            self.rsvp.first_stim_time,
            self.static_offset,
            buf_length=self.trial_length)

        # filter it
        notch_filterted_data = notch.notch_filter(
            raw_data, self.fs, self.notch_filter_frequency)
        filtered_data = bandpass.butter_bandpass_filter(
            notch_filterted_data, self.filter_low, self.filter_high, self.fs, order=self.filter_order)
        data = downsample.downsample(
            filtered_data, factor=self.downsample_rate)
        letters, times, target_info = self.letter_info(triggers, target_info)

        # reshape with the filtered data with our desired window length
        reshaped_data, _, _, _ = trial_reshaper(
            target_info,
            times,
            data,
            fs=self.fs,
            k=self.downsample_rate, mode='calibration',
            channel_map=self.channel_map,
            trial_length=self.trial_length)
        return reshaped_data

    def letter_info(self, triggers: List[Tuple[str, float]],
                    target_info: List[str]
                    ) -> Tuple[List[str], List[float], List[str]]:
        """
        Filters out non-letters and separates timings from letters.
        Parameters:
        -----------
         triggers: triggers e.g. [['A', 0.5], ...]
                as letter and flash time for the letter
         target_info: target information about the stimuli;
            ex. ['nontarget', 'nontarget', ...]
        Returns:
        --------
            (letters, times, target_info)
        """
        letters = []
        times = []
        target_types = []

        for i, (letter, stamp) in enumerate(triggers):
            if letter not in self.nonletters:
                letters.append(letter)
                times.append(stamp)
                target_types.append(target_info[i])

        # Raise an error if the stimuli includes unexpected terms
        if not set(letters).issubset(self.valid_targets):
            invalid = set(letters).difference(self.valid_targets)
            raise Exception(
                f'unexpected letters received: {invalid}')

        return letters, times, target_types

    @classmethod
    def label(cls):
        return RSVPInterSequenceFeedbackCalibration.TASK_NAME

    def name(self):
        return RSVPInterSequenceFeedbackCalibration.TASK_NAME
