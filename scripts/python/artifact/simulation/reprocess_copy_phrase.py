"""
Reprocesses the copy phrase using a new trained model.

Assumes the following directory structure:

USERS/
    USER_ID/
        CAL/
            user_id_dir/
                <condition_prefix>_model_*auc*.pkl
        NP_*order*/
            CPfolder1/
                raw_data.csv
                session.json
                triggers.txt
                ...
            CPfolder2/
                ...
            CPfolder3/
                ...
            CPfolder4/
                ...

It will do the following:

For each user:
    For each condition_prefix:
        1. Load a model from the CAL/ directory.
        2. Loop through the No Preview Copy Phrase Sessions
        3. Use the model to determine likelihood/probability of each inquiry
        4. Save the results to a file.
"""
from pathlib import Path
import pickle
import logging
import numpy as np
import json
import time

from bcipy.config import DEFAULT_DEVICE_SPEC_FILENAME, RAW_DATA_FILENAME, TRIGGER_FILENAME, DEFAULT_PARAMETERS_PATH
from bcipy.helpers.stimuli import grouper, update_inquiry_timing
from bcipy.helpers.load import load_experimental_data, load_raw_data, load_json_parameters
from bcipy.helpers.triggers import trigger_decoder, TriggerType
from bcipy.helpers.stimuli import InquiryReshaper, alphabet
from bcipy.acquisition import devices
from bcipy.helpers.raw_data import RawData
from bcipy.helpers.acquisition import analysis_channels
from bcipy.signal.process import filter_inquiries, get_default_transform, ERPTransformParams
from bcipy.helpers.parameters import Parameters
from bcipy.signal.model import SignalModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Constants
CONDITION_PREFIXES = ['of', 'cf']
SYMBOL_SET = alphabet()

def load_model(condition_prefix: str, session_path: Path) -> SignalModel:
    """Load the pkl model for the given condition_prefix"""
    # load the model with the prefix (e.g. of_model_0.1234.pkl)
    # find a model with the prefix
    for path in session_path.iterdir():
        if path.is_dir():
            response = next(path.glob(f"{condition_prefix}_model_*.pkl"), None)
            if response and condition_prefix in response.name:
                model_path = response
                logger.info(f"Found model for {condition_prefix} in {model_path}")
                break
            else:
                raise FileNotFoundError(f"Could not find model for {condition_prefix} in {session_path}")   
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_data(data_folder: Path, parameters: Parameters) -> tuple:
    # This data uses trial length and NOT trial window
    trial_length = parameters.get("trial_length", 0.5)
    trials_per_inquiry = parameters.get("stim_length")
    prestim_length = parameters.get("prestim_length", trial_length)
    buffer_length = int(parameters.get("task_buffer_length") / 2)
    # get signal filtering information
    static_offset = parameters.get("static_trigger_offset", 0.1)
    transform_params: ERPTransformParams = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate

    logger.info(
        f"\nData processing settings: \n"
        f"{str(transform_params)} \n"
        f"Trial Length: {trial_length}s, Trials per Inquiry: {trials_per_inquiry} \n"
        f"Prestimulus Buffer: {prestim_length}s, Poststimulus Buffer: {buffer_length}s \n"
        f"Static offset: {static_offset}"
    )

    # load the raw data
    raw_data: RawData = load_raw_data(f"{data_folder}/{RAW_DATA_FILENAME}.csv")
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
        device_type="EEG",
        remove_pre_fixation=True,
    )

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
    return trials, trigger_symbols, trials_per_inquiry, inquiries, inquiry_labels

def get_model_results(
        model: SignalModel,
        trials: np.ndarray,
        trigger_symbols: np.ndarray,
        trials_per_inquiry: int,
        inquiries: np.ndarray,
        inquiry_labels: np.ndarray,
        symbol_set: list,
        counter) -> dict:
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

        outputs[i + counter] = {
            "eeg_likelihood_evidence": list(response),
            "target_idx": target_index_in_alphabet,
            "nontarget_idx": nontarget_idx_in_alphabet,
        }
    return outputs


def main(directory: Path, parameters: Parameters) -> dict:
    """Main Analysis Loop.
    
    Assumes the data structure documented above.
    """
    results = {}
    # loop over users
    for user_path in directory.iterdir():

        # skip non-directories
        if not user_path.is_dir():
            continue
        
        # get the user_id
        user_id = user_path.name
        results[user_id] = {}
        print(f"Processing user: {user_id}")

        # loop over condition prefixes
        for condition_prefix in CONDITION_PREFIXES:
            # load the model
            results[user_id][condition_prefix] = {}
            model = load_model(condition_prefix, directory / user_id / "CAL")
            # loop over the NP folders
            data_folders = user_path.glob("*P*")
            if not data_folders:
                logger.warning(f"No data folders found for {user_id}")
                break

            inquiry_counter = 0
            results[user_id][condition_prefix]['model'] = {}
            for np_folders in data_folders:
                for np_folder in np_folders.iterdir():
                    if not np_folder.is_dir():
                        continue

                    # # load the data
                    trials, trigger_symbols, trials_per_inquiry, inquiries, inquiry_labels = load_data(np_folder, parameters)
                    # get the model results
                    outputs = get_model_results(
                        model,
                        trials,
                        trigger_symbols,
                        trials_per_inquiry,
                        inquiries,
                        inquiry_labels,
                        SYMBOL_SET,
                        counter=inquiry_counter)
                    results[user_id][condition_prefix]['model'].update(outputs)
                    inquiry_counter += len(outputs)
                    print(f"Processed {np_folder.name}")
                print(f"Processed {condition_prefix} for {user_id}. Total inquiries: {inquiry_counter}")
                # get estimated # of selections using the output, where the decision threshold is 0.8
            selections = estimate_selections_from_output(results[user_id][condition_prefix]['model'])
            results[user_id][condition_prefix]['selections'] = selections
    # save the results dict
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    with open(f"reprocess_all_copy_phrase_{timestamp}.json", "w") as f:
        json.dump(results, f)

    return results

def estimate_selections_from_output(output: dict, threshold: float = 0.8) -> list:
    """Estimate the number of selections based on the output."""
    total_selections = 0
    previous_probs = None
    for inquiry in output:
        # covert the likelihoods to probabilities
        likelihoods = output[inquiry]['eeg_likelihood_evidence']

        # if we have a previous probability, use it via multiplication
        if previous_probs is not None:
            likelihoods = np.multiply(likelihoods, previous_probs)
        probabilities = np.exp(likelihoods) / (1 + np.exp(likelihoods))

        # get the number of selections
        selections = np.sum(probabilities > threshold)

        # a selection
        if selections > 0:
            previous_probs = None
            total_selections += 1
        else:
            previous_probs = probabilities

    return total_selections


if __name__ in "__main__":
    # Load all users
    logger.info("Pipeline starting...")
    all_users = load_experimental_data()
    parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
    try:
        results = main(Path(all_users), parameters)
    except Exception as e:
        breakpoint()
        print(f"Error processing users: {e}")
        raise e
    
    logger.info("Processing complete!")
    breakpoint()
