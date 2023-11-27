import json
import logging
import os
from pathlib import Path

from bcipy.config import (
    RAW_DATA_FILENAME,
    DEFAULT_DEVICE_SPEC_FILENAME,
    TRIGGER_FILENAME)

from bcipy.helpers.acquisition import analysis_channels
import bcipy.acquisition.devices as devices
import mne

from bcipy.helpers.stimuli import TrialReshaper, InquiryReshaper, update_inquiry_timing, mne_epochs
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.signal.process import filter_inquiries, get_default_transform, get_fir_transform
from bcipy.helpers.load import load_raw_data, load_json_parameters
from bcipy.helpers.convert import convert_to_mne

import numpy as np
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

def load_data_inquiries(
        data_folder: Path,
        trial_length=None,
        pre_stim=0.0,
        apply_filter=True):
    """Loads raw data, and performs preprocessing by notch filtering, bandpass filtering, and downsampling.

    Args:
        data_folder (Path): path to raw data in BciPy format
        trial_length (float): length of each trial in seconds
        pre_stim_offset (float): window of time before stimulus onset to include in analysis

    Returns:
        raw_data: raw data object
        trial_data: np.ndarray: data, shape (trials, channels, time)
        labels: np.ndarray: labels, shape (trials,)
        trigger_timing: list of trigger timings
        channel_map: list of channels used in analysis
        poststim_length: float: length of each trial in seconds
        default_transform: transform used to filter data
        drop_log: dict: number of dropped trials

    """
    # Load parameters
    parameters = load_json_parameters(Path(data_folder, "parameters.json"), value_cast=True)
    poststim_length = trial_length if trial_length is not None else parameters.get("trial_length")
    pre_stim = pre_stim if pre_stim > 0.0 else parameters.get("prestim_length")

    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    log.info(
        f"\nData processing settings: \n"
        f"Filter: [{filter_low}-{filter_high}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {pre_stim}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )

    # default_transform = get_fir_transform(
    #     sample_rate_hz=sample_rate,
    #     notch_freq_hz=notch_filter,
    #     low=filter_low,
    #     high=filter_high,
    #     fir_design='firwin',
    #     fir_window='hamming',
    #     phase='zero-double',
    #     downsample_factor=downsample_rate,
    # )

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {device_spec}, fs={sample_rate}")

     # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    log.info(f'Channels used in analysis: {channels_used}')
    data, fs = raw_data.by_channel()
    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=data,
        sample_rate=fs,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=poststim_length,
        prestimulus_length=pre_stim,
        transformation_buffer=buffer,
    )

    if apply_filter:
        inquiries, fs = filter_inquiries(inquiries, default_transform, fs)
    inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
    trial_duration_samples = int(poststim_length * fs)
    trial_data = InquiryReshaper().extract_trials(
        inquiries, trial_duration_samples, inquiry_timing)

    # define the training classes using integers, where 0=nontargets/1=targets
    labels = inquiry_labels.flatten()
    trial_data = np.transpose(trial_data, (1, 0, 2)) # (epochs, channels, samples)
    drop_log = {
        'nontarget': labels.tolist().count(0),
        'target': labels.tolist().count(1),
        'nontarget_orig': trigger_targetness.count('nontarget'),
        'target_orig': trigger_targetness.count('target')}

    return raw_data, trial_data, labels, trigger_timing, channel_map, poststim_length, default_transform, drop_log, channels_used


def load_data_mne(
        data_folder,
        mne_data_annotations=None,
        trial_length=None,
        pre_stim=0.0,
        drop_artifacts=False,
        parameters=None):
    """Loads raw data, filters using default transform with parameters, and reshapes into trials."""
    # Load parameters
    parameters = parameters if parameters else load_json_parameters(Path(data_folder, "parameters.json"), value_cast=True)
    poststim_length = trial_length if trial_length is not None else parameters.get("trial_length")
    pre_stim = pre_stim if pre_stim > 0.0 else parameters.get("prestim_length")

    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    log.info(
        f"\nData processing settings: \n"
        f"Filter: [{filter_low}-{filter_high}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {pre_stim}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

    # # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )

    # default_transform = get_fir_transform(
    #     sample_rate_hz=sample_rate,
    #     notch_freq_hz=notch_filter,
    #     low=filter_low,
    #     high=filter_high,
    #     fir_design='firwin',
    #     fir_window='hamming',
    #     phase='zero-double',
    #     downsample_factor=downsample_rate,
    # )

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {device_spec}, fs={sample_rate}")

     # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    log.info(f'Channels used in analysis: {channels_used}')

    # Load data into MNE
    mne_data, fs = convert_to_mne(raw_data, channel_map, transform=default_transform, volts=False)

    if mne_data_annotations is not None:
        mne_data.set_annotations(mne_data_annotations)
    # remove any ignore annotations from the mn_data
    mne_data.annotations.delete(mne_data.annotations.description == 'ignore')

    trigger_labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]
    epochs = mne_epochs(
        mne_data,
        trigger_timing,
        trigger_labels,
        poststim_length,
        baseline=None,
        reject_by_annotation=drop_artifacts)
    
    # TODO use the epoch drop log? write to the session? Then we can use it via the inquiry based method
    
    labels = []
    for i in range(len(epochs)):
        try:
            epochs[i].event_id['1']
            labels.append(0)
        except:
            labels.append(1)

    nontarget = labels.count(0)
    target = labels.count(1)

    nontarget_orig = trigger_targetness.count('nontarget')
    target_orig = trigger_targetness.count('target')
    print(f"nontarget: {nontarget}, target: {target}")

    # if nontarget < 500 or target < 50:
    #     log.info("Not enough data to train model")
    #     raise Exception("Not enough data to train model")
    
    # put back into BciPy format
    trial_data = epochs.get_data(units='uV', tmin=0, tmax=poststim_length) # (epochs, channels, samples)
    drop_log = {'nontarget': nontarget, 'target': target, 'nontarget_orig': nontarget_orig, 'target_orig': target_orig}

    return (
        raw_data,
        trial_data,
        labels,
        trigger_timing,
        channel_map,
        poststim_length,
        default_transform,
        drop_log,
        epochs
    )

def load_data_trials(data_folder: Path, trial_length=None, pre_stim=0.0):
    """Loads raw data, and performs preprocessing by notch filtering, bandpass filtering, and downsampling.

    Args:
        data_folder (Path): path to raw data in BciPy format
        trial_length (float): length of each trial in seconds
        pre_stim_offset (float): window of time before stimulus onset to include in analysis

    Returns:
        np.ndarray: data, shape (trials, channels, time)
        np.ndarray: labels, shape (trials,)
        int: sampling rate (Hz)
    """
    # Load parameters
    parameters = load_json_parameters(Path(data_folder, "parameters.json"), value_cast=True)
    poststim_length = trial_length if trial_length is not None else parameters.get("trial_length")
    pre_stim = pre_stim if pre_stim > 0.0 else parameters.get("prestim_length")

    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    log.info(
        f"\nData processing settings: \n"
        f"Filter: [{filter_low}-{filter_high}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {pre_stim}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {device_spec}, fs={sample_rate}")

     # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    data, fs = raw_data.by_channel(transform=default_transform)
    trial_data, labels = TrialReshaper()(
        trigger_targetness,
        trigger_timing,
        data,
        fs,
        channel_map=channel_map,
        poststimulus_length=poststim_length)
    
    trial_data = np.transpose(trial_data, (1, 0, 2)) # (epochs, channels, samples)

    return raw_data, trial_data, labels, trigger_timing, channel_map, poststim_length, default_transform
