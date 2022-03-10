import numpy as np
from pathlib import Path

from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.task import InquiryReshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform
from loguru import logger


def get_trials_from_inquiries(inquiries, inquiry_labels):
    breakpoint()


def load_model(model_path, k_folds):
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.load(model_path)
    return model


def main(data_folder, parameters, model_path: Path, output_path: Path):

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    triggers_file = parameters.get("trigger_file_name", "triggers")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    hp_filter = parameters.get("filter_high")
    lp_filter = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")

    # get offset and k folds
    static_offset = parameters.get("static_trigger_offset")
    k_folds = parameters.get("k_folds")

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    # default_transform = get_default_transform(
    #     sample_rate_hz=fs,
    #     notch_freq_hz=notch_filter,
    #     bandpass_low=lp_filter,
    #     bandpass_high=hp_filter,
    #     bandpass_order=filter_order,
    #     downsample_factor=downsample_rate,
    # )
    # data, fs = default_transform(raw_data.by_channel(), fs)
    data = raw_data.by_channel()

    # Process triggers.txt
    trigger_values, trigger_timing, _ = trigger_decoder(
        offset=static_offset, trigger_path=f"{data_folder}/{triggers_file}.txt"
    )

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    model = load_model(model_path, k_folds)

    inquiries, inquiry_labels = InquiryReshaper()(
        trial_labels=trigger_values,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        trial_length=trial_length,
    )

    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    inquiries, fs = default_transform(inquiries, fs)
    trials, trial_labels = get_trials_from_inquiries(inquiries, inquiry_labels)

    auc = model.evaluate(trials, trial_labels)
    logger.info(f"AUC: {auc}")

    likelihood_updates = []
    for trial, label in zip(trials, trial_labels):
        likelihood_updates.append(model.predict(trial, label))

    np.save(output_path, np.array(likelihood_updates))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    parser.add_argument("-p", "--parameters_file", default="bcipy/parameters/parameters.json")
    parser.add_argument("-m", "--model_file", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    logger.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    main(args.data_folder, parameters, args.model_file, args.output_path)
    logger.info("Offline Analysis complete.")
