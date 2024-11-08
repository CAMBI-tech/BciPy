"""Script that will replay sessions and allow us to simulate new model predictions on that data."""
import json
import logging as logger
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import bcipy.acquisition.devices as devices
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME,
                          DEFAULT_PARAMETERS_FILENAME, RAW_DATA_FILENAME,
                          SESSION_DATA_FILENAME, TRIGGER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.list import grouper
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.stimuli import InquiryReshaper, update_inquiry_timing
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.signal.model import PcaRdaKdeModel, SignalModel
from bcipy.signal.process import (ERPTransformParams, filter_inquiries,
                                  get_default_transform)

logger.getLogger().setLevel(logger.INFO)


def load_model(model_path: Path, k_folds: int, model_class=PcaRdaKdeModel):
    """Load the model at the given path"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def generate_replay_outputs(
        data_folder,
        parameters,
        model_path: Path,
        model_class=PcaRdaKdeModel,
        symbol_set=alphabet(),
        write_output=False) -> Tuple[dict, list, list]:
    """Replay a session and generate outputs for the provided model.

    Parameters
    ----------
    data_folder: Path
        Path to the data folder containing the session data (session.json, raw_data.csv, triggers.txt)
    parameters: dict
        Parameters to use for the replay session.
    model_path: Path
        Path to the model to use for the replay session.
    model_class: class
        Class of the model to use for the replay session. By default, this is PcaRdaKdeModel.
    symbol_set: list
        List of symbols to use for the replay session. By default, this is the alphabet.
    write_output: bool
        Whether or not to write the output to a file. By default, this is False.

    Returns
    -------
    tuple - new_model_outputs, old_model_target_output, old_model_nontarget_output
    """
    k_folds = parameters.get("k_folds")
    model: SignalModel = load_model(model_path, k_folds, model_class)
    logger.info(f"Loaded model from {model_path}")

    # get trial information; to make backwards compatible, we will try to get the trial length
    # from the parameters file (old), but if it's not there, we use the trial window and adjust timing (>2.0).
    adjust_trials_by_window = False
    trial_length = parameters.get("trial_length", None)
    if trial_length is None:
        trial_window = parameters.get("trial_window")
        trial_length = trial_window[1] - trial_window[0]
        adjust_trials_by_window = True
        logger.info(f"Trial Window: {trial_window[0]}-{trial_window[1]}s")

    trials_per_inquiry = parameters.get("stim_length")
    prestim_length = parameters.get("prestim_length", trial_length)
    buffer_length = int(parameters.get("task_buffer_length") / 2)

    # get signal filtering information
    static_offset = parameters.get("static_trigger_offset")
    k_folds = parameters.get("k_folds")
    transform_params = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate

    logger.info(
        f"\nData processing settings: \n"
        f"{str(transform_params)} \n"
        f"Trial Length: {trial_length}s, Trials per Inquiry: {trials_per_inquiry} \n"
        f"Prestimulus Buffer: {prestim_length}s, Poststimulus Buffer: {buffer_length}s \n"
        f"Static offset: {static_offset}"
    )

    data_list = load_raw_data(data_folder, [f"{RAW_DATA_FILENAME}.csv"])
    # NOTE: With the current inputs this function only loads the EEG data. Update needed for eyetracker data.
    raw_data = data_list[0]
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate
    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)
    channel_map = analysis_channels(channels, device_spec)

    # extract the raw data for the channels we care about. The channel map will further reduce this if necessary.
    # by default, only the sample number and internal trigger channels are excluded.
    raw_data, _ = raw_data.by_channel()

    logger.info(f"Device type: {type_amp}")
    logger.info(f"Channels read from csv: {channels}")
    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    logger.info(f'Channels used in analysis: {channels_used}')

    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    # Adjust trigger timing by trial window if necessary
    if adjust_trials_by_window:
        trigger_timing = [timing + trial_window[0] for timing in trigger_timing]

    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=raw_data,
        sample_rate=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer_length,
    )
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=transform_params.notch_filter_frequency,
        bandpass_low=transform_params.filter_low,
        bandpass_high=transform_params.filter_high,
        bandpass_order=transform_params.filter_order,
        downsample_factor=transform_params.down_sampling_rate,
    )

    inquiries, transformed_sample_rate = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(trial_length * transformed_sample_rate)
    inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
    trials = InquiryReshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    # get the model outputs using the reshaped data
    outputs = {}
    inquiry_worth_of_trials = np.split(trials, inquiries.shape[1], 1)
    inquiry_worth_of_letters = grouper(trigger_symbols, trials_per_inquiry, incomplete="ignore")
    for i, (inquiry_trials, this_inquiry_letters, this_inquiry_labels) in enumerate(
        zip(inquiry_worth_of_trials, inquiry_worth_of_letters, inquiry_labels)
    ):
        response = model.compute_likelihood_ratio(inquiry_trials, this_inquiry_letters, symbol_set=symbol_set)
        if np.any(this_inquiry_labels == 1):
            index_of_target_trial = np.argwhere(this_inquiry_labels == 1)[0][0]
            target_letter = this_inquiry_letters[index_of_target_trial]
            target_index_in_alphabet = symbol_set.index(target_letter)

            nontarget_idx_in_alphabet = [symbol_set.index(q) for q in this_inquiry_letters if q != target_letter]
        else:
            target_index_in_alphabet = None
            nontarget_idx_in_alphabet = [symbol_set.index(q) for q in this_inquiry_letters]

        outputs[i] = {
            "eeg_evidence": list(response),
            "target_idx": target_index_in_alphabet,
            "nontarget_idx": nontarget_idx_in_alphabet,
        }

    if write_output:
        with open(write_output / "replay_outputs.json", "w") as f:
            json.dump(outputs, f, indent=2)

    # Get values computed during actual experiment from session.json for comparison
    session_json = data_folder / SESSION_DATA_FILENAME
    all_target_eeg, all_nontarget_eeg = load_eeg_evidence_from_session_json(session_json, symbol_set)

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


def load_eeg_evidence_from_session_json(session_json, symbol_set) -> Tuple[list, list]:
    """Load EEG evidence from session.json file for comparison with replay outputs.

    Parameters
    ----------
    session_json : str
        Path to session.json file
    symbol_set : list
        List of symbols used in experiment

    Returns
    -------
    all_target_eeg : list
        List of EEG evidence for target stimuli
    all_nontarget_eeg : list
        List of EEG evidence for nontarget stimuli
    """
    with open(session_json, "r") as f:
        contents = json.load(f)
    series = contents["series"]

    all_target_eeg = []
    all_nontarget_eeg = []
    for inquiries in series:
        for inquiry in inquiries:
            # There are cases in which an inquiry may have other evidence types but no ERP. Ex. InquiryPreview
            if len(inquiry.evidences[EvidenceType.ERP]) < 1:
                print("Skipping Inquiry. No ERP evidence.")
                continue
            else:
                stim_label = inquiry["stimuli"][0]  # name of symbols presented
                stim_label.pop(0)  # remove fixation cross
                stim_indices = [symbol_set.index(sym) for sym in stim_label]
                targetness = inquiry["target_info"]  # targetness of stimuli
                targetness.pop(0)  # remove fixation cross
                target = [index for index, label in zip(stim_indices, targetness) if label == "target"]
                nontarget = [index for index, label in zip(stim_indices, targetness) if label == "nontarget"]
                all_target_eeg.extend([inquiry.evidences[EvidenceType.ERP][pos] for pos in target])
                all_nontarget_eeg.extend([inquiry.evidences[EvidenceType.ERP][pos] for pos in nontarget])

    return all_target_eeg, all_nontarget_eeg


def plot_collected_outputs(
        outputs_with_new_model: dict,
        targets_with_old_model: list,
        nontargets_with_old_model: list,
        outdir: str) -> None:
    """Plot collected outputs from replay experiment.

    Parameters
    ----------
    outputs_with_new_model : dict
        List of outputs from replay experiment using new model
    targets_with_old_model : list
        List of outputs from the session data using old model for target stimuli
    nontargets_with_old_model : list
        List of outputs from the session using old model for nontarget stimuli
    outdir : str
        Path to directory to save plots
    """
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
    parser.add_argument(
        "-d",
        "--data_folders",
        action="append",
        type=Path,
        required=True,
        help="Session data folders to be processed. This argument can be repeated to accumulate sessions data.")
    parser.add_argument("-m", "--model_file", required=True)
    parser.add_argument("-o", "--outdir", type=Path, default=None)
    parser.add_argument(
        "-p",
        "--parameter_file",
        type=Path,
        default=None,
        help="Parameter file to be used for replay. If none, the session parameter file will be used.")

    args = parser.parse_args()

    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = Path(__file__).resolve().parent
        print(f"Outdir: {outdir}")
    
    data_folders = args.data_folders
    model_file = args.model_file

    if args.outdir is None:
        args.outdir = Path(__file__).resolve().parent

    logger.info(f"Outdir: {args.outdir}")

    outputs_with_new_model = []
    targets_with_old_model = []
    nontargets_with_old_model = []
    for data_folder in args.data_folders:
        logger.info(f"Processing {data_folder}")

        if args.parameter_file is not None:
            params_file = args.parameter_file
        else:
            params_file = Path(data_folder, DEFAULT_PARAMETERS_FILENAME)
        logger.info(f"Loading params from {params_file}")
        params = load_json_parameters(params_file, value_cast=True)

        # Generate replay outputs using the model provided against the session data in data_folder
        outputs, all_target_eeg, all_nontarget_eeg = generate_replay_outputs(
            data_folder, params, args.model_file, write_output=False
        )
        outputs_with_new_model.append(outputs)
        targets_with_old_model.extend(all_target_eeg)
        nontargets_with_old_model.extend(all_nontarget_eeg)

    plot_collected_outputs(
        outputs_with_new_model,
        targets_with_old_model,
        nontargets_with_old_model,
        args.outdir)

    # breakpoint()

    logger.info("Replay complete.")