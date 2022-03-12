from asyncio import new_event_loop
from random import sample
import numpy as np
from pathlib import Path
from itertools import zip_longest

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


def get_trials_from_inquiries(inquiries, samples_per_trial, inquiry_timing, downsample_rate):
    new_trials = []
    num_inquiries = inquiries.shape[1]
    for inquiry_idx, timing in zip(range(num_inquiries), inquiry_timing): # C x I x S

        for time in timing:
            time = time // downsample_rate
            y = time + samples_per_trial
            new_trials.append(inquiries[:,inquiry_idx,time:y])
    return np.stack(new_trials, 1) # C x T x S

    


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
    inquiries, inquiry_labels, inquiry_timing = q
    ()(
        trial_targetness_label=trigger_targetness,
        trial_stimuli_label=trigger_labels,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        trial_length=trial_length,
    )
    
    # old_shape = inquiries.shape # (C, I, 699)
    # logger.info(f'Old Shape: {old_shape}')
    # inq_flatten = inquiries.reshape(-1, old_shape[-1])
    # logger.info(f'Inq Flatten I: {inq_flatten.shape}')
    # inq_flatten_filtered, fs = default_transform(inq_flatten, fs)
    # logger.info(f'Inq Flatten Filtered: {inq_flatten_filtered.shape}')
    # inquiries = inq_flatten_filtered.reshape(*old_shape[:2], inq_flatten_filtered.shape[-1])
    trial_duration_samples = int(trial_length * fs)
    # next_trial_samples = int(time_flash * fs)
    trials = get_trials_from_inquiries(
        inquiries, trial_duration_samples, inquiry_timing, downsample_rate)

            
    logger.info(np.all(np.isfinite(trials)))
    logger.info(f'Inquiries Final: {inquiries.shape}')
    logger.info(f'Trials Extracted: {trials.shape}')

    # auc = model.evaluate(trials, trial_labels)
    # logger.info(f"AUC: {auc}")

    likelihood_updates = []
    inquiry_worth_of_trials = np.split(trials, inquiries.shape[1], 1)
    inquiry_worth_of_labels = grouper(trigger_labels, trials_per_inquiry, incomplete='ignore')
    
    for inquiry_trials, inquiry_labels in zip(inquiry_worth_of_trials, inquiry_worth_of_labels):
        # model.predict(trials[:][0], trigger_labels[:len(trials[:][0])])
        import pdb; pdb.set_trace()
        # inquiry_trials = np.array(inquiry_trials)
        inquiry_labels = list(inquiry_labels)
        response = model.predict(inquiry_trials, inquiry_labels, symbol_set=alphabet()) # Note the first trial is the same always!

        import pdb; pdb.set_trace()

    # np.save(output_path, np.array(response))

    return

"""
Replay Full File Filter

Replay Inquiry Based Filtering
"""


def validate_inquiry_based_trials_against_trial_reshaper(new_trials, trials):
    """Add np.allclose(new_trials, trials)"""
    pass


def plot_eeg():
    """TODO: use the MNE snippets?"""
    pass

"""

# logger.info(f"Likelihood: {response}")
        # import pdb; pdb.set_trace()
    # likelihood_updates.append(response)
    # except Exception as e:
    #     data, fs = default_transform(raw_data.by_channel(), 300)
    #     # because the reshapers can change timing with offsets, we should still return the timing that updated
    #     new_trials, targetness_labels = model.reshaper(
    #             trial_targetness_label=trigger_targetness,
    #             # trial_stimuli_label=trigger_labels,
    #             timing_info=trigger_timing,
    #             eeg_data=data,
    #             fs=fs,
    #             trials_per_inquiry=trials_per_inquiry,
    #             channel_map=channel_map,
    #             trial_length=trial_length,
    #         )
    #     response = model.evaluate(new_trials, targetness_labels)
    #     response_trials_from_inq = model.evaluate(trials, targetness_labels)
    #     response = model.predict(new_trials, trigger_labels, symbol_set=alphabet())
    #     response2 = model.predict(trials, trigger_labels, symbol_set=alphabet())
"""

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
