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
    DEVICE_SPEC_PATH,
)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.task.data import EvidenceType
import bcipy.acquisition.devices as devices
from bcipy.helpers.list import grouper
from bcipy.helpers.load import load_json_parameters, load_raw_data, load_experimental_data, load_signal_model_path
from bcipy.helpers.stimuli import InquiryReshaper, update_inquiry_timing
from bcipy.helpers.session import read_session
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform, filter_inquiries

from scipy import stats

SYMBOLS = alphabet()


def load_model(model_path: Path, k_folds: int):
    """Load the PcaRdaKdeModel model at the given path"""
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.load(model_path)
    return model


def generate_replay_outputs(data_folder, parameters, model_path: Path, write_output=None):
    """Generate Replay outputs.

    This function will load the raw data, filter it, and compare the model predictions to the
    session data evidence output.

    It can be used to compare two distinct models (a new model trained given model_path),
    or the same model used to generate the session.json using different parameters (e.g. filters).

    It will:
        1. Load model from model_path
        2. Load session data from data_folder
        3. Compare model predictions to session data from older model
        4. Write output to file if write_output is True

    raw data -> inquiries -> filter -> trials -> run model
    """
    session_json = data_folder / SESSION_DATA_FILENAME
    # check if session.json exists at the given path
    if not Path(session_json).exists():
        raise FileNotFoundError(f"Session data file not found at {session_json}")
    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    prestim_length = parameters.get("prestim_length", trial_length)
    buffer = int(parameters.get("task_buffer_length") / 2)
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

    devices.load(Path(DEVICE_SPEC_PATH))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

    print(f"Channels read from csv: {channels}")
    print(f"Device type: {type_amp}")

    # Process triggers.txt
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    print(f'Channels used in analysis: {channels_used}')

    model = load_model(model_path, k_folds)
    data, fs = raw_data.by_channel()

    print(
        f"\nData processing settings: \n"
        f"Filter: [{lp_filter}-{hp_filter}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {trial_length}s, Prestimulus: {prestim_length}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=data,
        sample_rate=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,  # use this to add time to the end of the Inquiry for processing!
    )
    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    inquiries, transformed_sample_rate = filter_inquiries(inquiries, default_transform, fs)
    trial_duration_samples = int(trial_length * transformed_sample_rate)
    inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
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
        with open(write_output / "replay_outputs.json", "w") as f:
            json.dump(outputs, f, indent=2)

    # Get values computed during actual experiment from session.json
    all_target_eeg, all_nontarget_eeg = load_from_session_json(session_json)
    

    # Check that the number of trials in triggers.txt matches the number of trials in session.json
    print("Length of trials from triggers.txt: ", len(trigger_timing))
    print("Length of all_target_eeg from session.json: ", len(all_target_eeg))
    print("Length of all_nontarget_eeg from session.json: ", len(all_nontarget_eeg))
    all_target_from_outputs = len([o["target_idx"] for o in outputs.values() if o["target_idx"] is not None])
    print("Length of all_target_from_outputs: ", all_target_from_outputs)
    all_nontarget_from_outputs = sum([len(o["nontarget_idx"]) for o in outputs.values()])
    print("Length of all_nontarget_from_outputs: ", all_nontarget_from_outputs)
    
    # Fail the replay if the number of trials in triggers.txt does not match the number of trials in session.json
    assert len(trigger_timing) == len(all_target_eeg) + len(all_nontarget_eeg), (
        "Length of trigger_timing does not match length of all_target_eeg + all_nontarget_eeg")
    assert len(trigger_timing) == all_target_from_outputs + all_nontarget_from_outputs, (
        "Length of trigger_timing does not match length of all_target_from_outputs + all_nontarget_from_outputs")

    return outputs, all_target_eeg, all_nontarget_eeg


def load_from_session_json(session_json) -> list:
    contents = read_session(session_json)
    series = contents.series

    all_target_eeg = []
    all_nontarget_eeg = []
    for inquiries in series:
        for inquiry in inquiries:
            # There are cases in which an inquiry may have other evidence types but no ERP. Ex. InquiryPreview
            if len(inquiry.evidences[EvidenceType.ERP]) < 1:
                print("Skipping Inquiry. No ERP evidence.")
                continue
            else:
                # get stimuli and targetness, remove fixation cross from first position
                stim_label = inquiry.stimuli
                stim_label.pop(0)
                stim_indices = [SYMBOLS.index(sym) for sym in stim_label]
                targetness = inquiry.target_info
                targetness.pop(0)

                # get indices of target and nontarget stimuli
                target = [index for index, label in zip(stim_indices, targetness) if label == "target"]
                nontarget = [index for index, label in zip(stim_indices, targetness) if label == "nontarget"]
                all_target_eeg.extend([inquiry.evidences[EvidenceType.ERP][pos] for pos in target])
                all_nontarget_eeg.extend([inquiry.evidences[EvidenceType.ERP][pos] for pos in nontarget])

    return all_target_eeg, all_nontarget_eeg


def plot_collected_outputs(outputs_with_new_model, targets_with_old_model, nontargets_with_old_model, outdir):
    records = []
    for output in outputs_with_new_model:  # each session
        for _, inquiry_contents in output.items():  # each inquiry
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
    print(f"{df.groupby('which_model').describe()}")
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
    plt.close()

    # return all target and nontarget responses
    # return df[df["which_model"] == "new_target"].values, df[df["which_model"] == "new_nontarget"].values
    return df[df["which_model"] == "new_target"]["response_value"].values, df[df["which_model"] == "new_nontarget"]["response_value"].values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folders", type=Path, default=None, help="Data folder(s): session.json must be present in each subfolder.")
    parser.add_argument("-m", "--model_file", default=None)
    parser.add_argument("-o", "--outdir", default=None, type=Path)
    parser.add_argument("-p", "--parameter_file", default=None, type=Path)


    args = parser.parse_args()

    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = Path(__file__).resolve().parent
        print(f"Outdir: {outdir}")
    
    data_folders = args.data_folders
    model_file = args.model_file

    if data_folders is None:
        data_folders = Path(load_experimental_data())

    if model_file is None:
        model_file = Path(load_signal_model_path())

    outputs_with_new_model = []
    targets_with_old_model = []
    nontargets_with_old_model = []
    for data_folder in data_folders.iterdir():
        print(f"Processing {data_folder}")
        try:
            # update this line to change processing parameters
            if args.parameter_file is not None:
                params_file = args.parameter_file
            else:
                params_file = Path(data_folder, DEFAULT_PARAMETER_FILENAME)
            print(f"Loading params from {params_file}")
            params = load_json_parameters(params_file, value_cast=True)
            outputs, all_target_eeg, all_nontarget_eeg = generate_replay_outputs(
                data_folder, params, model_file, write_output=outdir
            )
            outputs_with_new_model.append(outputs)
            targets_with_old_model.extend(all_target_eeg)
            nontargets_with_old_model.extend(all_nontarget_eeg)
        except Exception as e:
            print(f"Failed to process {data_folder}. Skipping. Error: {e}")


    # compare the outputs of the new model with the outputs of the old model visually
    new_targets, new_nontargets = plot_collected_outputs(outputs_with_new_model, targets_with_old_model, nontargets_with_old_model, outdir)

    # compare the outputs of the new model with the outputs of the old model statistically
    print("Comparing the outputs of the new model with the outputs of the old model statistically")
    target_diffs = new_targets - np.array(targets_with_old_model)
    nontarget_diffs = new_nontargets - np.array(nontargets_with_old_model)

    normal_targets = stats.normaltest(target_diffs).pvalue > 0.05
    print(f"Target diffs are normally distributed? {normal_targets}")
    normal_nontargets = stats.normaltest(nontarget_diffs).pvalue > 0.05
    print(f"Nontarget diffs are normally distributed? {normal_nontargets}")

    # if not normal, set equal_var=False to use Welch's t-test
    stats_target = stats.ttest_ind(targets_with_old_model, new_targets.tolist(), equal_var=normal_targets)
    print(
        f"The older model produced targets that were {stats_target.statistic} times as large as new target. "
        f"Is this a significant difference? {stats_target.pvalue < 0.05} {stats_target.pvalue}")
    stats_nontarget = stats.ttest_ind(nontargets_with_old_model, new_nontargets.tolist(), equal_var=normal_nontargets)
    print(
        f"The older model produced nontargets that were {stats_nontarget.statistic} times as large as new nontarget. "
        f"Is this a significant difference? {stats_nontarget.pvalue < 0.05} {stats_nontarget.pvalue}")

    print("Replay complete.")
