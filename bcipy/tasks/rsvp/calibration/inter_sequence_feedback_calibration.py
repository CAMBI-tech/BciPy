from psychopy import core
from typing import List, Tuple

from bcipy.feedback.visual.level_feedback import LevelFeedback
from bcipy.tasks.task import Task
from bcipy.tasks.rsvp.calibration.calibration import RSVPCalibrationTask

from bcipy.helpers.triggers import _write_triggers_from_sequence_calibration
from bcipy.helpers.stimuli_generation import random_rsvp_calibration_seq_gen, get_task_info
from bcipy.signal.processing.sig_pro import sig_pro
from bcipy.helpers.bci_task_related import (
    calculate_stimulation_freq,
    trial_complete_message,
    trial_reshaper,
    get_user_input,
    pause_calibration,
    process_data_for_decision)

from bcipy.helpers.acquisition_related import analysis_channels, analysis_channels_by_device

from bcipy.signal.processing.decomposition.psd import power_spectral_density, PSD_TYPE


class RSVPInterSequenceFeedbackCalibration(Task):
    """RSVP InterSequenceFeedbackCalibration Task uses inter sequence
        feedback to alert user to their current state in order to increase performance
        in a calibration task.

    Calibration task performs an RSVP stimulus sequence to elicit an ERP.
    Parameters will change how many stim and for how long they present.
    Parameters also change color and text / image inputs.

    Note: The channel index, PSD method, and stimulation frequency band
     approximation are hardcoded for piloting. In the future, these
     may be changed to allow easier feedback for different caps.


    Input:
        win (PsychoPy Display Object)
        daq (Data Acquisition Object)
        parameters (Dictionary)
        file_save (String)

    Output:
        file_save (String)
    """
    TASK_NAME = 'RSVP Inter Sequence Feedback Calibration Task'

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
        self.num_sti = self._task.num_sti
        self.len_sti = self._task.len_sti
        self.is_txt_sti = self.rsvp.is_txt_sti
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

        self.feedback_buffer_time = self.parameters['feedback_buffer_time']
        self.feedback_line_color = self.parameters['feedback_line_color']
        self.time_flash = self.parameters['time_flash']
        self.stimulation_frequency = calculate_stimulation_freq(self.time_flash)

        # get +/- 10% of the stimulation frequency
        self.psd_export_band = (
            self.stimulation_frequency * .90,
            self.stimulation_frequency * 1.1)

        self.trial_length = self.time_flash * self.len_sti
        self.k = self.parameters['down_sampling_rate']
        self.filtered_sampling_rate = self.fs / self.k
        self.psd_method = PSD_TYPE.WELCH

        # The channel used to calculate the SSVEP response to RSVP sequence.
        # NOTE: This task will only work for VR300
        self.psd_channel_index = 6
        self.device_name = self.daq.device_info.name
        self.channel_map = analysis_channels(self.daq.device_info.channels, self.device_name)

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
                 num_sti=self.num_sti,
                 len_sti=self.len_sti,
                 timing=self.timing,
                 is_txt=self.is_txt_sti,
                 color=self.color)

            (task_text, task_color) = get_task_info(self.num_sti,
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
                self.rsvp.stim_sequence = stimuli_elements[sequence_idx]

                # check if text stimuli or not for color information
                if self.is_txt_sti:
                    self.rsvp.color_list_sti = color_sti[sequence_idx]

                self.rsvp.time_list_sti = timing_sti[sequence_idx]

                # Wait for a time
                core.wait(self._task.buffer_val)

                # Do the sequence
                last_sequence_timing = self.rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, self._task.trigger_file)

                position = self._get_feedback_decision(last_sequence_timing)
                timing = self.visual_feedback.administer(position=position)

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
        #   tranisiton less abrupt to the user
        core.wait(self.trial_length + self.feedback_buffer_time)

        # get data sequence
        data = self._get_data_for_psd(sequence_timing)

        # we always want the same data channel in the occipital region and the first of it
        response = power_spectral_density(
            data[self.psd_channel_index][0],
            self.psd_export_band,
            sampling_rate=self.filtered_sampling_rate,
            # plot=True,  # uncomment to see the PSD plot in real time
            method=self.psd_method)

        # In the event the calcalated band returns nothing, throw an
        #  error
        if response == 0:
            raise ValueError('PSD calcualted for feedback invalid')

        # TODO: finish this approzimation of feedback position with Barry
        if response > 20:
            return 5
        elif response > 10:
            return 3
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
        filtered_data = sig_pro(raw_data, fs=self.fs, k=self.k)
        letters, times, target_info = self.letter_info(triggers, target_info)

        # reshape with the filtered data with our desired window length
        data, _, _, _ = trial_reshaper(
            target_info,
            times,
            filtered_data,
            fs=self.fs,
            k=self.k, mode='copy_phrase',
            channel_map=self.channel_map,
            trial_length=self.trial_length)
        return data

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
                f'unexpected letters received in copy phrase: {invalid}')

        return letters, times, target_types

    @classmethod
    def label(cls):
        return RSVPInterSequenceFeedbackCalibration.TASK_NAME

    def name(self):
        return RSVPInterSequenceFeedbackCalibration.TASK_NAME
