import json
import logging as logger
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bcipy.config import (
    RAW_DATA_FILENAME,
    TRIGGER_FILENAME,
    DEFAULT_PARAMETER_FILENAME, SESSION_DATA_FILENAME,
    DEFAULT_DEVICE_SPEC_FILENAME,
)
from bcipy.helpers.acquisition import analysis_channels
import bcipy.acquisition.devices as devices
from bcipy.helpers.list import grouper
from bcipy.helpers.load import load_json_parameters, load_raw_data, load_experimental_data
from bcipy.helpers.session import read_session, evidence_records
from bcipy.helpers.stimuli import update_inquiry_timing
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform, filter_inquiries, ERPTransformParams

logger.getLogger().setLevel(logger.INFO)
log = logger.getLogger(__name__)


@dataclass()
class ExtractedExperimentData:  # TODO clean up design
    inquiries: np.ndarray
    trials: np.ndarray
    labels: list
    inquiry_timing: list

    decoded_triggers: tuple


def process_raw_data_for_model(data_folder, parameters, model_class=PcaRdaKdeModel) -> ExtractedExperimentData:
    assert parameters, "Parameters are required for offline analysis."
    if not data_folder:
        data_folder = load_experimental_data()

    # extract relevant session information from parameters file
    trial_window = parameters.get("trial_window")
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

    log.info(
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

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {type_amp}, fs={sample_rate}")

    k_folds = parameters.get("k_folds")
    model = model_class(k_folds=k_folds)

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
    log.info(f'Channels used in analysis: {channels_used}')

    data, fs = raw_data.by_channel()

    inquiries, inquiry_labels, inquiry_timing = model.reshaper(
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
    trials = model.reshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    # define the training classes using integers, where 0=nontargets/1=targets
    # labels = inquiry_labels.flatten()

    return ExtractedExperimentData(inquiries, trials, inquiry_labels, inquiry_timing, (trigger_targetness, trigger_timing, trigger_symbols))
