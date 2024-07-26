# This script is to process matrix and rsvp calibrations together to determine if they can be used together.
#  It requires bcipy and pyriemann to be installed. `pip install -e .` at root of project and `pip install pyriemann`
import logging
from pathlib import Path
from bcipy.helpers.load import load_json_parameters, load_raw_data, load_experimental_data
from bcipy.helpers.triggers import trigger_decoder, TriggerType
from bcipy.config import (
    DEFAULT_PARAMETER_FILENAME,
    RAW_DATA_FILENAME,
    TRIGGER_FILENAME,
    DEFAULT_DEVICE_SPEC_FILENAME)
from bcipy.helpers.stimuli import update_inquiry_timing
from bcipy.signal.process import get_default_transform, filter_inquiries
from bcipy.acquisition import devices
from bcipy.helpers.stimuli import InquiryReshaper
from bcipy.helpers.acquisition import analysis_channels
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")


# TODO: only take the first half of the data for calibrations to simulate the bcifit experiment?


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
    parameters = load_json_parameters(Path(data_folder, DEFAULT_PARAMETER_FILENAME), value_cast=True)
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

    return trial_data, labels


if __name__ in "__main__":
    from grid_search import crossvalidate_record
    results = {}
    trial_length = 0.5
    # load matrix calibration
    print("Please select matrix calibration data")
    matrix_calib_path = load_experimental_data()
    trial_data_matrix, labels_matrix = load_data_inquiries(matrix_calib_path, trial_length=trial_length)

    # load rsvp calibration
    print("Please select RSVP calibration data")
    rsvp_calib_path = load_experimental_data()
    trial_data_rsvp, labels_rsvp = load_data_inquiries(rsvp_calib_path, trial_length=trial_length)

    # combine the two datasets
    trial_data = np.concatenate((trial_data_matrix, trial_data_rsvp), axis=0)
    labels = np.concatenate((labels_matrix, labels_rsvp), axis=0)

    print(f"Matrix calibration data shape: {trial_data_matrix.shape}, RSVP calibration data shape: {trial_data_rsvp.shape}")

    # print(f"Combined data shape: {trial_data.shape}, Combined labels shape: {labels.shape}")

    # cross validate the combined data
    response, scores = crossvalidate_record((trial_data, labels))
    for name in scores:
        results[name] = response[f'mean_test_{name}']
        results[f'std_{name}'] = response[f'std_test_{name}']

    print(results)
    print("Done")

    breakpoint()

    