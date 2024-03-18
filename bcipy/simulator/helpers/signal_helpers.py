import logging as logger
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from bcipy.acquisition import devices
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME, RAW_DATA_FILENAME,
                          TRIGGER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.exceptions import TaskConfigurationException
from bcipy.helpers.list import grouper
from bcipy.helpers.load import load_raw_data
from bcipy.helpers.stimuli import InquiryReshaper, update_inquiry_timing
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.signal.process import (ERPTransformParams, filter_inquiries,
                                  get_default_transform)

log = logger.getLogger(__name__)


@dataclass()
class ExtractedExperimentData:
    source_dir: str
    inquiries: np.ndarray
    trials: np.ndarray
    labels: List
    inquiry_timing: List

    decoded_triggers: tuple
    trials_per_inquiry: int

    @property
    def trials_by_inquiry(self) -> List[np.ndarray]:
        """EEG data by inquiry.
        shape (i_inquiry, n_channel, m_trial, x_sample)
        """
        return np.split(self.trials, self.inquiries.shape[1], 1)

    @property
    def symbols_by_inquiry(self) -> List[List[str]]:
        """Get symbols by inquiry. shape (i_inquiry, s_alphabet_subset)"""
        _targets, _times, trigger_symbols = self.decoded_triggers
        return [
            list(group) for group in grouper(
                trigger_symbols, self.trials_per_inquiry, incomplete="ignore")
        ]

    @property
    def labels_by_inquiry(self) -> List[List[int]]:
        """Shape (i_inquiry, s_alphabet_subset)"""
        return self.labels


def process_raw_data_for_model(data_folder, parameters,
                               reshaper: InquiryReshaper = InquiryReshaper()) -> ExtractedExperimentData:
    assert parameters, "Parameters are required for offline analysis."

    log.debug(f"Processing raw data for {data_folder}")
    # extract relevant session information from parameters file
    trial_window = parameters.get("trial_window")
    if trial_window is None:
        # using trial_length instead and converting to trial window
        trial_length = parameters.get('trial_length')
        if not trial_length:
            raise TaskConfigurationException(
                "Could not find trial_window or trial_length in parameters")

        trial_window = tuple([0, trial_length])  # trial_length=0.5 -> trial_window=(0,0.5)

    window_length = trial_window[1] - trial_window[0]
    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    # get signal filtering information
    transform_params = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate
    static_offset = parameters.get("static_trigger_offset")

    log.debug(
        f"\nData processing settings: \n"
        f"{str(transform_params)} \n"
        f"Trial Window: {trial_window[0]}-{trial_window[1]}s, "
        f"Prestimulus Buffer: {prestim_length}s, Poststimulus Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=transform_params.notch_filter_frequency,
        bandpass_low=transform_params.filter_low,
        bandpass_high=transform_params.filter_high,
        bandpass_order=transform_params.filter_order,
        downsample_factor=transform_params.down_sampling_rate,
    )

    log.debug(f"Channels read from csv: {channels}")
    log.debug(f"Device type: {type_amp}, fs={sample_rate}")

    # Process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    # update the trigger timing list to account for the initial trial window
    corrected_trigger_timing = [timing + trial_window[0] for timing in trigger_timing]

    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    log.debug(f'Channels used in analysis: {channels_used}')

    data, fs = raw_data.by_channel()

    inquiries, inquiry_labels, inquiry_timing = reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=corrected_trigger_timing,
        eeg_data=data,
        sample_rate=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=window_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,
    )

    inquiries, fs = filter_inquiries(inquiries, default_transform, sample_rate)
    inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
    trial_duration_samples = int(window_length * fs)
    trials = reshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    # define the training classes using integers, where 0=nontargets/1=targets
    # labels = inquiry_labels.flatten()

    return ExtractedExperimentData(
        data_folder,
        inquiries,
        trials,
        inquiry_labels,
        inquiry_timing,
        decoded_triggers=(trigger_targetness, trigger_timing, trigger_symbols),
        trials_per_inquiry=trials_per_inquiry)
