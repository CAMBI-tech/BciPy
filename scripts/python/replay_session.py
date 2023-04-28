"""Script that will replay sessions and allow us to simulate new model predictions on that data."""
import json
import logging as logger
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
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.stimuli import InquiryReshaper, TrialReshaper, update_inquiry_timing
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform

logger.getLogger()

SYMBOLS = alphabet()


def load_model(model_path: Path, k_folds: int):
    """Load the PcaRdaKdeModel model at the given path"""
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.load(model_path)
    return model


def comparison1(data_folder, parameters, model_path: Path):
    """Check that we get trials from correct positions"""

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    prestim_length = parameters.get("prestim_length", trial_length)
    # time_flash = parameters.get("time_flash")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    hp_filter = parameters.get("filter_high")
    lp_filter = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")
    k_folds = parameters.get("k_folds")

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, f"{RAW_DATA_FILENAME}.csv"))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

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
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, device_spec)

    model = load_model(model_path, k_folds)

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, _, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=data,
        sample_rate=transformed_sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        prestimulus_length=prestim_length,
        transformation_buffer=trial_length,  # use this to add time to the end of the Inquiry for processing!
    )

    # NOTE -
    # If doing: Raw -> filter -> inquiries -> InquiryReshaper.extract_trials
    # Then do: use downsample_rate=1 during InquiryReshaper.extract_trials
    # If doing:  Raw -> inquiries -> filter -> InquiryReshaper.extract_trials
    # Then do: use downsample_rate=2 during InquiryReshaper.extract_trials

    # Uncomment to filter by inquiries
    # inquiries = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(trial_length * transformed_sample_rate)
    inquiry_timing = update_inquiry_timing(inquiry_timing, transformed_sample_rate)
    trials = InquiryReshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    # "Can we get the trials from exact same time windows"
    # 1a:   raw data -> filtered data -> TrialReshaper
    # 1b:   raw data -> filtered -> InquiryReshaper -> TrialReshaper
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
    assert validate_response


def generate_replay_outputs(data_folder, parameters, model_path: Path, write_output=False):
    """Try running a previous model as follows. Shouldn't get errors.

    # 2a (real, online model): raw -> inquiries -> filter -> trials -> model preds
    # 2b (trying to replay):  raw data -> InquiryReshaper -> filtered -> TrialReshaper

    raw data -> inquiries -> filter -> trials -> run model
    """

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    prestim_length = parameters.get("prestim_length", trial_length)
    # time_flash = parameters.get("time_flash")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    hp_filter = parameters.get("filter_high")
    lp_filter = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")
    k_folds = parameters.get("k_folds")

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, f"{RAW_DATA_FILENAME}.csv"))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    # Process triggers.txt
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, device_spec)

    model = load_model(model_path, k_folds)

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=raw_data.by_channel(),
        fs=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        prestimulus_length=prestim_length,
        transformation_buffer=trial_length,  # use this to add time to the end of the Inquiry for processing!
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
    inquiry_timing = update_inquiry_timing(inquiry_timing, transformed_sample_rate)
    trials = InquiryReshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    alpha = alphabet()
    outputs = {}
    inquiry_worth_of_trials = np.split(trials, inquiries.shape[1], 1)
    inquiry_worth_of_letters = grouper(trigger_symbols, trials_per_inquiry, incomplete="ignore")
    for i, (inquiry_trials, this_inquiry_letters, this_inquiry_labels) in enumerate(
        zip(inquiry_worth_of_trials, inquiry_worth_of_letters, inquiry_labels)
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

        outputs[i] = {
            "eeg_evidence": list(response),
            "target_idx": target_index_in_alphabet,
            "nontarget_idx": nontarget_idx_in_alphabet,
        }

    if write_output:
        with open(data_folder / "replay_outputs.json", "w") as f:
            json.dump(outputs, f, indent=2)

    # Get values computed during actual experiment from session.json
    session_json = data_folder / SESSION_DATA_FILENAME
    all_target_eeg, all_nontarget_eeg = load_from_session_json(session_json)

    return outputs, all_target_eeg, all_nontarget_eeg


def load_from_session_json(session_json) -> list:
    with open(session_json, "r") as f:
        contents = json.load(f)
    series = contents["series"]

    all_target_eeg = []
    all_nontarget_eeg = []

    for inquiries in series.values():
        for inquiry in inquiries.values():
            if len(inquiry["eeg_evidence"]) < 1:
                continue
            else:
                stim_label = inquiry["stimuli"][0]  # name of symbols presented
                stim_label.pop(0)
                stim_indices = [SYMBOLS.index(sym) for sym in stim_label]
                targetness = inquiry["target_info"]  # targetness of stimuli
                # target_letter = inquiry['target_letter']
                target = [index for index, label in zip(stim_indices, targetness) if label == "target"]
                nontarget = [index for index, label in zip(stim_indices, targetness) if label == "nontarget"]
                all_target_eeg.extend([inquiry["eeg_evidence"][pos] for pos in target])
                all_nontarget_eeg.extend([inquiry["eeg_evidence"][pos] for pos in nontarget])

    return all_target_eeg, all_nontarget_eeg


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
    trials, _targetness_labels = TrialReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=data,
        sample_rate=sample_rate,
        channel_map=channel_map,
        poststimulus_length=trial_length,
    )

    logger.info(f"is finite (trials): {np.all(np.isfinite(trials))}")
    logger.info(f"is finite (inquiry_trials): {np.all(np.isfinite(inquiry_trials))}")
    is_allclose = np.allclose(trials, inquiry_trials)
    logger.info(f"Is allclose: {is_allclose}")
    return is_allclose


def filter_inquiries(inquiries, transform, sample_rate):
    old_shape = inquiries.shape  # (C, I, 699)
    inq_flatten = inquiries.reshape(-1, old_shape[-1])  # (C*I, 699)
    inq_flatten_filtered, transformed_sample_rate = transform(inq_flatten, sample_rate)
    inquiries = inq_flatten_filtered.reshape(*old_shape[:2], inq_flatten_filtered.shape[-1])  # (C, I, ...)
    return inquiries, transformed_sample_rate


def plot_collected_outputs(outputs_with_new_model, targets_with_old_model, nontargets_with_old_model, outdir):
    records = []
    for output in outputs_with_new_model:  # each session
        for inquiry_idx, inquiry_contents in output.items():  # each inquiry
            target_idx = inquiry_contents["target_idx"]
            if target_idx is not None:
                records.append(
                    {
                        "which_model": "new_target",
                        "response_value": inquiry_contents["eeg_evidence"][target_idx],
                    }
                )

            nontarget_idx = inquiry_contents["nontarget_idx"]
            for i in nontarget_idx:
                records.append({"which_model": "new_nontarget", "response_value": inquiry_contents["eeg_evidence"][i]})

    for target_response in targets_with_old_model:
        records.append({"which_model": "old_target", "response_value": target_response})
    for nontarget_response in nontargets_with_old_model:
        records.append({"which_model": "old_nontarget", "response_value": nontarget_response})

    df = pd.DataFrame.from_records(records)
    logger.info(f"{df.describe()}")
    ax = sns.stripplot(
        x="which_model",
        y="response_value",
        data=df,
        order=["old_target", "new_target", "old_nontarget", "new_nontarget"],
    )
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"color": "k", "ls": "-", "lw": 2},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        x="which_model",
        y="response_value",
        data=df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
        order=["old_target", "new_target", "old_nontarget", "new_nontarget"],
    )

    ax.set(yscale="log")
    plt.savefig(outdir / "response_values.stripplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    ax = sns.boxplot(
        x="which_model",
        y="response_value",
        data=df,
        order=["old_target", "new_target", "old_nontarget", "new_nontarget"],
    )
    ax.set(yscale="log")
    plt.savefig(outdir / "response_values.boxplot.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folders", action="append", type=Path, default=None)
    parser.add_argument("-m", "--model_file", required=True)
    outdir = Path(__file__).resolve().parent
    logger.info(f"Outdir: {outdir}")
    args = parser.parse_args()

    assert len(set(args.data_folders)) == len(args.data_folders), "Duplicated data folders"

    outputs_with_new_model = []
    targets_with_old_model = []
    nontargets_with_old_model = []
    for data_folder in args.data_folders:
        logger.info(f"Processing {data_folder}")
        params_file = Path(data_folder, DEFAULT_PARAMETER_FILENAME)
        logger.info(f"Loading params from {params_file}")
        params = load_json_parameters(params_file, value_cast=True)
        comparison1(data_folder, params, args.model_file)
        outputs, all_target_eeg, all_nontarget_eeg = generate_replay_outputs(
            data_folder, params, args.model_file, write_output=True
        )
        outputs_with_new_model.append(outputs)
        targets_with_old_model.extend(all_target_eeg)
        nontargets_with_old_model.extend(all_nontarget_eeg)

    plot_collected_outputs(outputs_with_new_model, targets_with_old_model, nontargets_with_old_model, outdir)

    logger.info("Replay complete.")
