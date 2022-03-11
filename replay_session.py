from random import sample
import numpy as np
from pathlib import Path

from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.task import alphabet, InquiryReshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform
from loguru import logger


def get_trials_from_inquiries(inquiries, samples_per_trial, next_trial_samples, trials_per_inquiry):
    new_trials = []
    for i in range(len(inquiries[0])):
        x = 0
        y = samples_per_trial
        for _ in range(trials_per_inquiry):
            new_trials.append(inquiries[:,i,x:y])
            x += next_trial_samples
            y += next_trial_samples
    
    labels = ['nontarget'] * len(new_trials)
    return np.stack(new_trials, 1), labels
    


def load_model(model_path, k_folds):
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.load(model_path)
    return model


def main(data_folder, parameters, model_path: Path, output_path: Path):
    """Main.
    
    Useful for determining impact of changing model parameters or type on previously collected data.
    """

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    time_flash = parameters.get("time_flash")
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
    trial_duration_samples = int(trial_length * fs)
    next_trial_samples = int(time_flash * fs)
    trials, trial_labels = get_trials_from_inquiries(
        inquiries, trial_duration_samples, next_trial_samples, trials_per_inquiry)

    # auc = model.evaluate(trials, trial_labels)
    # logger.info(f"AUC: {auc}")

    likelihood_updates = []
    for trial, label in zip(trials, trial_labels):
        response = model.predict(trial, label, symbol_set=alphabet())
        logger.info(f"Likelihood: {response}")
        likelihood_updates.append(response)

    np.save(output_path, np.array(likelihood_updates))

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    # parser.add_argument("-p", "--parameters_file", default="bcipy/parameters/parameters.json")
    parser.add_argument("-m", "--model_file", required=True)
    # parser.add_argument("-o", "--output_path", required=False)
    args = parser.parse_args()

    parameters = f'{args.data_folder}/parameters.json'
    logger.info(f"Loading params from {parameters}")
    parameters = load_json_parameters(parameters, value_cast=True)
    main(args.data_folder, parameters, args.model_file, args.data_folder)
    logger.info("Offline Analysis complete.")
