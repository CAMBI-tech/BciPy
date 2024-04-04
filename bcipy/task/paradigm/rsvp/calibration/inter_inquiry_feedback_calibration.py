from typing import List, NamedTuple, Tuple

import numpy as np
from psychopy import core

from bcipy.acquisition.multimodal import ContentType
from bcipy.feedback.visual.level_feedback import LevelFeedback
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.stimuli import TrialReshaper
from bcipy.helpers.task import get_device_data_for_decision, relative_triggers
from bcipy.signal.process.decomposition.psd import (PSD_TYPE,
                                                    power_spectral_density)
from bcipy.signal.process.transform import (ERPTransformParams,
                                            get_default_transform)
from bcipy.task.exceptions import InsufficientDataException
from bcipy.task.paradigm.rsvp.calibration.calibration import \
    RSVPCalibrationTask


class FeedbackConfig(NamedTuple):
    """Configuration parameters for feedback calculation and display.
    Note: filter parameters are modified in the parameters.json file.
    """
    feedback_buffer_time: int = 1  # seconds

    # Method used to approximate power spectral density bands (Welch or MultiTaper)
    psd_method: PSD_TYPE = PSD_TYPE.WELCH
    # The channel used to calculate the PSD from an inquiry.
    # We want to use a posterior channel. If Oz available, use that!
    # Note that the index is 0-based and should be the index in the list of analysis_channels.
    # TODO: use channel name
    psd_channel_index: int = 5
    psd_channel_name: str = 'Oz'

    psd_lower_limit = 8  # power spectral density band lower bound (in Hz)
    psd_upper_limit = 10  # power spectral density band upper bound (in Hz)

    # Feedback level thresholds (as a relative PSD value)
    lvl_5_threshold: float = 0.2052
    lvl_4_threshold: float = 0.2693
    lvl_3_threshold: float = 0.3147
    lvl_2_threshold: float = 0.3780

    # true/false order is desceding from 5 -> 1 for level
    feedback_descending: bool = False

    @property
    def psd_export_band(self):
        """psd band of interest to use for feeback (low, high)"""
        return (self.psd_lower_limit, self.psd_upper_limit)


def determine_feedback_response(config: FeedbackConfig,
                                response: float) -> int:
    """Determine feedback response.

    Depending on the band chosen to give feedback off, we may need to invert the
        levels. By default, it's in descending order. Set feedback_level_descending
        to false for ascending
    """
    # default condition; for use with SSVEP or PSD that increases with focus to task
    if config.feedback_descending:
        if response > config.lvl_5_threshold:
            return 5
        if response > config.lvl_4_threshold:
            return 4
        if response > config.lvl_3_threshold:
            return 3
        if response > config.lvl_2_threshold:
            return 2
        return 1

    # ascending condition; use with PSD that decrease with focus to task
    else:
        if response < config.lvl_5_threshold:
            return 5
        if response < config.lvl_4_threshold:
            return 4
        if response < config.lvl_3_threshold:
            return 3
        if response < config.lvl_2_threshold:
            return 2
        return 1


class RSVPInterInquiryFeedbackCalibration(RSVPCalibrationTask):
    """RSVP InterInquiryFeedbackCalibration Task uses inter inquiry
        feedback to alert user to their current state in order to increase performance
        in a calibration task.

    Calibration task performs an RSVP stimulus inquiry to elicit an ERP.
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
    TASK_NAME = 'RSVP Inter inquiry Feedback Calibration Task'

    def __init__(self, win, daq, parameters, file_save):
        super().__init__(win, daq, parameters, file_save)

        self.visual_feedback = LevelFeedback(display=self.window,
                                             parameters=self.parameters,
                                             clock=self.experiment_clock)

        self.nonletters = ['+', 'PLUS', 'calibration_trigger']
        self.valid_targets = self.alp

        self.time_flash = self.parameters['time_flash']
        self.downsample_rate = self.parameters['down_sampling_rate']

        eeg_spec = self.daq.clients_by_type.get(ContentType.EEG).device_spec
        self.sample_rate = eeg_spec.sample_rate
        self.channel_map = analysis_channels(eeg_spec.channels, eeg_spec)

        self.filtered_sampling_rate = self.sample_rate / self.downsample_rate
        # length of time to use for PSD calculation
        self.trial_length = self.time_flash * self.stim_length

        transform_params = parameters.instantiate(ERPTransformParams)
        self.transform = get_default_transform(
            sample_rate_hz=self.sample_rate,
            notch_freq_hz=transform_params.notch_filter_frequency,
            bandpass_low=transform_params.filter_low,
            bandpass_high=transform_params.filter_high,
            bandpass_order=transform_params.filter_order,
            downsample_factor=transform_params.down_sampling_rate,
        )
        self.reshaper = TrialReshaper()
        self.feedback_config = FeedbackConfig()

    def show_feedback(self, timing: List[Tuple[str, float]]) -> None:
        """Shows feedback after an inquiry. Called by the execute loop after
        writing triggers.
        
        Parameters
        ----------
            timing - list of (trigger_label, timestamp) pairs
        """
        self.logger.info('[Feedback] Getting Decision')
        # wait some time in order to get enough data from the daq and make the
        # transition less abrupt to the user
        core.wait(self.feedback_config.feedback_buffer_time)
        position = self._get_feedback_decision(timing)
        self.logger.info(
            f'[Feedback] Administering feedback position {position}')
        self.visual_feedback.administer(position=position)

    def _get_feedback_decision(self,
                               inquiry_timing: List[Tuple[str, float]]) -> int:
        """Make a decision from the data on which feedback item to highlight.

        Parameters
        ----------
           timing - list of (trigger_label, timestamp) pairs for triggers from
               the last presented inquiry. timestamp values are in experiment
               clock units.
        """

        # exclude target and fixation
        inquiry_of_interest = inquiry_timing[-self.stim_length:]

        # get data for inquiry and only use the first 2 stimuli
        data = self._get_data_for_psd(inquiry_of_interest[:2])

        # we always want the same data channel in the occipital region and the
        # first of it
        response = power_spectral_density(
            data[self.feedback_config.psd_channel_index][0],
            self.feedback_config.psd_export_band,
            sampling_rate=self.filtered_sampling_rate,
            # plot=True,  # uncomment to see the PSD plot in real time
            method=self.feedback_config.psd_method,
            relative=True)

        self.logger.info(f'[Feedback] Response decision {response}')

        # In the event the calcalated band returns nothing, throw an
        #  error
        if response == 0:
            message = 'PSD calculated for feedback invalid'
            self.logger.error(f'[Feedback] {message}')
            raise InsufficientDataException(message)

        return determine_feedback_response(self.feedback_config, response)

    def _get_data_for_psd(
            self, inquiry_timing: List[Tuple[str, float]]) -> np.ndarray:
        """Get EEG data from the data acquisition and reshape.
        
        Parameters
        ----------
            inquiry_timing - list of (trigger_label, timestamp) pairs for
                triggers from the last presented inquiry.
        Returns
        -------
            processed (transformed and reshaped) data associated with the
                given inquiry.
        """

        # currently prestim_length is used as a buffer for filter application
        post_stim_buffer = int(self.parameters.get('task_buffer_length') / 2)
        prestim_buffer: float = self.parameters['prestim_length']
        trial_window: Tuple[float, float] = self.parameters['trial_window']
        trial_length = trial_window[1] - trial_window[0]

        # filter only triggers of interest (alphabet letters and '+') and
        # update the inquiry timing list (stim, time) based on the trial window first time value
        inquiry_timing = [(stim, time + trial_window[0])
                          for stim, time in inquiry_timing
                          if stim in (self.alp + ['+'])]

        # gets raw data for all devices in the specified window of time.
        device_data = get_device_data_for_decision(
            inquiry_timing=inquiry_timing,
            daq=self.daq,
            offset=self.parameters['static_trigger_offset'],
            prestim=prestim_buffer,
            poststim=post_stim_buffer + trial_length)

        # we are only interested in EEG data
        raw_data = device_data.get(ContentType.EEG)
        transformed_data, transform_sample_rate = self.transform(
            raw_data, self.sample_rate)

        triggers = relative_triggers(inquiry_timing, prestim_buffer)
        times = letter_times(triggers,
                             nonletters=self.nonletters,
                             valid_targets=self.valid_targets)

        # reshape with the filtered data with our desired window length
        reshaped_data, _lbls = self.reshaper(
            trial_targetness_label=['nontarget'] * len(times),
            timing_info=times,
            eeg_data=transformed_data,
            sample_rate=transform_sample_rate,
            channel_map=self.channel_map,
            poststimulus_length=trial_length)
        
        return reshaped_data

    @classmethod
    def label(cls):
        return RSVPInterInquiryFeedbackCalibration.TASK_NAME

    def name(self):
        return RSVPInterInquiryFeedbackCalibration.TASK_NAME


def letter_times(
    triggers: List[Tuple[str, float]],
    nonletters: List[str] = None,
    valid_targets: List[str] = None
) -> Tuple[List[str], List[float], List[str]]:
    """
    Filters out non-letters and separates timings from letters.
    Parameters:
    -----------
        triggers: triggers e.g. [['A', 0.5], ...]
            as letter and flash time for the letter
       
    Returns:
    --------
        times
    """
    letters = []
    times = []

    for (letter, stamp) in triggers:
        if letter not in nonletters:
            letters.append(letter)
            times.append(stamp)

    # Raise an error if the stimuli includes unexpected terms
    if not set(letters).issubset(valid_targets):
        invalid = set(letters).difference(valid_targets)
        raise Exception(f'unexpected letters received: {invalid}')

    return times
