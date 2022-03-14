
from curses import raw
from pathlib import Path
from itertools import zip_longest

from loguru import logger
import numpy as np

from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.task import alphabet, InquiryReshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform


def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')

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
    static_offset = parameters.get("static_trigger_offset")
    k_folds = parameters.get("k_folds")

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    # data = raw_data.by_channel()
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
    trigger_targetness, trigger_timing, trigger_labels = trigger_decoder(
        offset=static_offset, trigger_path=f"{data_folder}/{triggers_file}.txt"
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    model = load_model(model_path, k_folds)

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        trial_stimuli_label=trigger_labels,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        transformation_buffer=0, # use this to add time to the end of the Inquiry for processing!
    )
    
    # Uncomment to filter by inquiries
    # inquiries = filter_inquiries(inquiries, default_transform, fs)
    trial_duration_samples = int(trial_length * fs)
    trials = InquiryReshaper.extract_trials(
        inquiries, trial_duration_samples, inquiry_timing, downsample_rate) # TODO not sure if we always need to account for downsample rate... that may be our issue


    # uncomment to validate against the trial reshaper. Note: check reshape parameters are set correctly!
    # validate_response = validate_inquiry_based_trials_against_trial_reshaper(raw_data, raw_data.sample_rate, model, default_transform, trigger_targetness, trigger_timing, trials_per_inquiry, channel_map, trial_length, trials)
    # logger.info(f'Inquiry trials == trial reshaper? {validate_response}')

            
    logger.info(np.all(np.isfinite(trials)))
    logger.info(f'Inquiries Final: {inquiries.shape}')
    logger.info(f'Trials Extracted: {trials.shape}')


    likelihood_updates = [] ## TODO: we eventually want to output a list of likelihoods per inquiry!!
    inquiry_worth_of_trials = np.split(trials, inquiries.shape[1], 1)
    inquiry_worth_of_labels = grouper(trigger_labels, trials_per_inquiry, incomplete='ignore')
    
    for inquiry_trials, inquiry_labels in zip(inquiry_worth_of_trials, inquiry_worth_of_labels):
        # model.predict(trials[:][0], trigger_labels[:len(trials[:][0])])
        # inquiry_trials = np.array(inquiry_trials)
        inquiry_labels = list(inquiry_labels)
        import pdb; pdb.set_trace()
        response = model.predict(inquiry_trials, inquiry_labels, symbol_set=alphabet()) # Note the first trial is the same always!

    # np.save(output_path, np.array(response))

    return likelihood_updates


def validate_inquiry_based_trials_against_trial_reshaper(
        raw_data,
        fs,
        model,
        transform,
        trigger_targetness,
        trigger_timing,
        trials_per_inquiry,
        channel_map,
        trial_length,
        inquiry_trials):
    """Add np.allclose(new_trials, trials)"""
    data, fs = transform(raw_data.by_channel(), fs)
    # because the reshapers can change timing with offsets, we should still return the timing that updated
    trials, targetness_labels = model.reshaper(
            trial_targetness_label=trigger_targetness,
            # trial_stimuli_label=trigger_labels,
            timing_info=trigger_timing,
            eeg_data=data,
            fs=fs,
            trials_per_inquiry=trials_per_inquiry,
            channel_map=channel_map,
            poststimulus_length=trial_length,
        )

    # uncomment to play around with model predictions
    # response = model.evaluate(trials, targetness_labels)
    # response_trials_from_inq = model.evaluate(inquiry_trials, targetness_labels)
    # response = model.predict(new_trials, trigger_labels, symbol_set=alphabet())
    # # response2 = model.predict(trials, trigger_labels, symbol_set=alphabet())
    np.all(np.isfinite(trials))
    np.all(np.isfinite(inquiry_trials))
    return np.allclose(trials, inquiry_trials)

def plot_eeg():
    """TODO: use the MNE snippets? See https://github.com/CAMBI-tech/BciPy/blob/ar_offline_analysis/bcipy/signal/model/ar_offline_analysis.py
    
    I've never passed MNE already trialed data, but the trigger timing etc generated from the various reshapers could be used after
     inputting the raw data to epoch and visualize. Sorry I didn't get further on this!
    """
    pass

def filter_inquiries(inquiries, transform, fs):
    old_shape = inquiries.shape # (C, I, 699)
    inq_flatten = inquiries.reshape(-1, old_shape[-1])
    inq_flatten_filtered, fs = transform(inq_flatten, fs)
    inquiries = inq_flatten_filtered.reshape(*old_shape[:2], inq_flatten_filtered.shape[-1])
    return inquiries


def get_trials_from_model_reshaper(raw_data, fs, model, transform, trigger_targetness, trigger_timing, trials_per_inquiry, channel_map, trial_length):
    data, fs = transform(raw_data.by_channel(), fs)
    # because the reshapers can change timing with offsets, we should still return the timing that updated
    trials, targetness_labels = model.reshaper(
            trial_targetness_label=trigger_targetness,
            # trial_stimuli_label=trigger_labels,
            timing_info=trigger_timing,
            eeg_data=data,
            fs=fs,
            trials_per_inquiry=trials_per_inquiry,
            channel_map=channel_map,
            poststimulus_length=trial_length,
        )
    # response = model.evaluate(new_trials, targetness_labels)
    # # response_trials_from_inq = model.evaluate(trials, targetness_labels)
    # response = model.predict(new_trials, trigger_labels, symbol_set=alphabet())
    # # response2 = model.predict(trials, trigger_labels, symbol_set=alphabet())
    return trials, targetness_labels, fs


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
