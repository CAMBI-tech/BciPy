"""Script that will replay sessions and allow us to simulate new model predictions on that data.

TODO Mar 15:
- Try to re-create the same outputs as in "session.json", as closely as possible.
  (load correct model, confirm using same filter params, confirm no changes about buffering window lengths, etc)
  Come up with simple way to decide "matching" or not.
- Load a "new" model trained with slightly modified code/parameters and replay.
- Cleanup this script and refactor as needed.
"""
import json
import logging
import sys
from bcipy.helpers.triggers import TriggerType

# from curses import raw
from itertools import zip_longest
from pathlib import Path

# from loguru import logger
import numpy as np

from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.task import InquiryReshaper, alphabet
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logger = logging.getLogger(__name__)


def grouper(iterable, chunk_size, *, incomplete="fill", fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    chunks = [iter(iterable)] * chunk_size
    if incomplete == "fill":
        return zip_longest(*chunks, fillvalue=fillvalue)
    if incomplete == "strict":
        return zip(*chunks, strict=True)
    if incomplete == "ignore":
        return zip(*chunks)

    raise ValueError("Expected fill, strict, or ignore")


def load_model(model_path: Path, k_folds: int):
    """Load the PcaRdaKdeModel model at the given path"""
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.load(model_path)
    return model


def comparison1(data_folder, parameters, model_path: Path, output_path: Path, write_output: bool = False):
    """Check that we get trials from correct positions"""

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    # time_flash = parameters.get("time_flash")
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
    sample_rate = raw_data.sample_rate

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    # data = raw_data.by_channel()
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    data, transformed_sample_rate = default_transform(raw_data.by_channel(), sample_rate)

    # Process triggers.txt
    trigger_targetness, trigger_timing, trigger_labels = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{triggers_file}.txt",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    model = load_model(model_path, k_folds)

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, _, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        trial_stimuli_label=trigger_labels,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=transformed_sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        transformation_buffer=0,  # use this to add time to the end of the Inquiry for processing!
    )

    # NOTE -
    # If doing: Raw -> filter -> inquiries -> InquiryReshaper.extract_trials
    # Then do: use downsample_rate=1 during InquiryReshaper.extract_trials
    # If doing:  Raw -> inquiries -> filter -> InquiryReshaper.extract_trials
    # Then do: use downsample_rate=2 during InquiryReshaper.extract_trials

    # Uncomment to filter by inquiries
    # inquiries = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(trial_length * transformed_sample_rate)
    trials = InquiryReshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    # "Can we get the trials from exact same time windows"
    # 1a:   raw data -> filtered data -> TrialReshaper
    # 1b:   raw data -> fitered -> InquiryReshaper -> TrialReshaper
    # Note that we're not doing the same kind of preprocessing
    # as we do "online".
    # uncomment to validate against the trial reshaper. Note: check reshape parameters are set correctly!
    validate_response = validate_inquiry_based_trials_against_trial_reshaper(
        raw_data,
        raw_data.sample_rate,
        model,
        default_transform,
        trigger_targetness,
        trigger_timing,
        trials_per_inquiry,
        channel_map,
        trial_length,
        trials,
    )
    logger.info(f"Inquiry trials == trial reshaper? {validate_response}")


def comparison2(data_folder, parameters, model_path: Path, output_path: Path, write_output: bool = False):
    """Try running a previous model as follows. Shouldn't get errors.

    # 2a (real, online model): raw -> inquiries -> filter -> trials -> model preds
    # 2b (trying to replay):  raw data -> InquiryReshaper -> filtered -> TrialReshaper

    raw data -> inquiries -> filter -> trials -> run model
    """

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    # time_flash = parameters.get("time_flash")
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
    sample_rate = raw_data.sample_rate

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    # Process triggers.txt
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{triggers_file}.txt",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    model = load_model(model_path, k_folds)

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        trial_stimuli_label=trigger_symbols,
        timing_info=trigger_timing,
        eeg_data=raw_data.by_channel(),
        fs=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        transformation_buffer=0,  # use this to add time to the end of the Inquiry for processing!
    )

    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    inquiries, transformed_sample_rate = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(trial_length * transformed_sample_rate)
    trials = InquiryReshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing, downsample_rate)

    alpha = alphabet()
    outputs = []  ## TODO: we eventually want to output a list of likelihoods per inquiry!!
    # target_likelihood_updates, nontarget_likelihood_updates = [], []
    inquiry_worth_of_trials = np.split(trials, inquiries.shape[1], 1)
    inquiry_worth_of_letters = grouper(trigger_symbols, trials_per_inquiry, incomplete="ignore")
    alphabet
    for inquiry_trials, this_inquiry_letters, this_inquiry_labels in zip(
        inquiry_worth_of_trials, inquiry_worth_of_letters, inquiry_labels
    ):
        response = model.predict(inquiry_trials, this_inquiry_letters, symbol_set=alpha)
        if np.any(this_inquiry_labels == 1):
            index_of_target_trial = np.argwhere(this_inquiry_labels == 1)[0][0]
            target_letter = this_inquiry_letters[index_of_target_trial]
            target_index_in_alphabet = alpha.index(target_letter)

            nontarget_idx_in_alphabet = [alpha.index(q) for q in this_inquiry_letters if q != target_letter]
        else:
            target_index_in_alphabet = None
            nontarget_idx_in_alphabet = [alpha.index(q) for q in this_inquiry_letters]

        outputs.append(
            {
                "likelihood_updates": list(response),
                "target_index": target_index_in_alphabet,
                "nontarget_idx": nontarget_idx_in_alphabet,
            }
        )

    if write_output:
        with open(output_path / "replay_outputs.json", "w") as f:
            json.dump(outputs, f, indent=2)

    return outputs


def validate_inquiry_based_trials_against_trial_reshaper(
    raw_data,
    sample_rate,
    model,
    transform,
    trigger_targetness,
    trigger_timing,
    trials_per_inquiry,
    channel_map,
    trial_length,
    inquiry_trials,
):
    """Add np.allclose(new_trials, trials)"""
    data, sample_rate = transform(raw_data.by_channel(), sample_rate)
    # because the reshapers can change timing with offsets, we should still return the timing that updated
    trials, _targetness_labels = model.reshaper(
        trial_targetness_label=trigger_targetness,
        # trial_stimuli_label=trigger_labels,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
    )

    # uncomment to play around with model predictions
    # response = model.evaluate(trials, targetness_labels)
    # response_trials_from_inq = model.evaluate(inquiry_trials, targetness_labels)
    # response = model.predict(new_trials, trigger_labels, symbol_set=alphabet())
    # # response2 = model.predict(trials, trigger_labels, symbol_set=alphabet())
    logger.info(f"is finite (trials): {np.all(np.isfinite(trials))}")
    logger.info(f"is finite (inquiry_trials): {np.all(np.isfinite(inquiry_trials))}")
    is_allclose = np.allclose(trials, inquiry_trials)
    logger.info(f"Is allclose: {is_allclose}")
    return is_allclose


def plot_eeg():
    """TODO: use the MNE snippets? See https://github.com/CAMBI-tech/BciPy/blob/ar_offline_analysis/bcipy/signal/model/ar_offline_analysis.py

    I've never passed MNE already trialed data, but the trigger timing etc generated from the
    various reshapers could be used after inputting the raw data to epoch and visualize. Sorry
    I didn't get further on this!
    """
    pass


def filter_inquiries(inquiries, transform, sample_rate):
    old_shape = inquiries.shape  # (C, I, 699)
    inq_flatten = inquiries.reshape(-1, old_shape[-1])  # (C*I, 699)
    inq_flatten_filtered, transformed_sample_rate = transform(inq_flatten, sample_rate)
    inquiries = inq_flatten_filtered.reshape(*old_shape[:2], inq_flatten_filtered.shape[-1])  # (C, I, ...)
    return inquiries, transformed_sample_rate


def get_trials_from_model_reshaper(
    raw_data,
    sample_rate,
    model,
    transform,
    trigger_targetness,
    trigger_timing,
    trials_per_inquiry,
    channel_map,
    trial_length,
):
    data, sample_rate = transform(raw_data.by_channel(), sample_rate)
    # because the reshapers can change timing with offsets, we should still return the timing that updated
    trials, targetness_labels = model.reshaper(
        trial_targetness_label=trigger_targetness,
        # trial_stimuli_label=trigger_labels,
        timing_info=trigger_timing,
        eeg_data=data,
        fs=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
    )
    # response = model.evaluate(new_trials, targetness_labels)
    # # response_trials_from_inq = model.evaluate(trials, targetness_labels)
    # response = model.predict(new_trials, trigger_labels, symbol_set=alphabet())
    # # response2 = model.predict(trials, trigger_labels, symbol_set=alphabet())
    return trials, targetness_labels, sample_rate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=Path, default=None)
    parser.add_argument("-m", "--model_file", required=True)
    args = parser.parse_args()

    params_file = Path(args.data_folder, "parameters.json")
    logger.info(f"Loading params from {params_file}")
    params = load_json_parameters(params_file, value_cast=True)
    comparison1(args.data_folder, params, args.model_file, args.data_folder)
    comparison2(args.data_folder, params, args.model_file, args.data_folder, write_output=True)
    logger.info("Offline Analysis complete.")
