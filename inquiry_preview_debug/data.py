from pathlib import Path

from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.signal.process import get_default_transform
from loguru import logger
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model import PcaRdaKdeModel


def load_data(input_path: Path):
    parameters_file = input_path / "parameters.json"
    parameters = load_json_parameters(parameters_file, value_cast=True)

    # extract relevant session information from parameters file
    trial_length = 0.5
    triggers_file = parameters.get("trigger_file_name", "triggers.txt")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate", 2)
    notch_filter = parameters.get("notch_filter_frequency", 60)
    hp_filter = parameters.get("filter_high", 45)
    lp_filter = parameters.get("filter_low", 2)
    filter_order = parameters.get("filter_order", 2)

    # get offset and k folds
    static_offset = parameters.get("static_trigger_offset", 0)

    # Load raw data
    raw_data = load_raw_data(Path(input_path, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    data, fs = default_transform(raw_data.by_channel(), fs)

    # Process triggers.txt
    _, t_t_i, t_i, offset = trigger_decoder(mode="calibration", trigger_path=f"{input_path}/{triggers_file}")

    offset = offset + static_offset

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    # channel_names = ["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz"]
    # channel_map = analysis_channels(channels, type_amp)
    channel_map = [0, 0, 1, 0, 1, 1, 1, 0]  # Same channels as alpha models are using
    data, labels = PcaRdaKdeModel.reshaper(
        trial_labels=t_t_i,
        timing_info=t_i,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=parameters.get("stim_length"),
        offset=offset,
        channel_map=channel_map,
        trial_length=trial_length,
    )

    return data, labels
