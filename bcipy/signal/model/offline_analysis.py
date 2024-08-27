# mypy: disable-error-code="attr-defined"
# needed for the ERPTransformParams
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import resample

import bcipy.acquisition.devices as devices
from bcipy.config import (BCIPY_ROOT, DEFAULT_DEVICE_SPEC_FILENAME,
                          DEFAULT_PARAMETERS_PATH, MATRIX_IMAGE_FILENAME,
                          STATIC_AUDIO_PATH, TRIGGER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels, raw_data_filename
from bcipy.helpers.load import (load_experimental_data, load_json_parameters,
                                load_raw_data)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import save_model
from bcipy.helpers.stimuli import play_sound, update_inquiry_timing
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.task import relative_triggers
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.visualization import (visualize_centralized_data,
                                         visualize_erp, visualize_gaze,
                                         visualize_gaze_accuracies,
                                         visualize_gaze_inquiries,
                                         visualize_results_all_symbols)
from bcipy.preferences import preferences
from bcipy.signal.model.base_model import SignalModel, SignalModelMetadata
from bcipy.signal.model.gaussian_mixture import (GMIndividual, GMCentralized,
                                                    KernelGP, KernelGPSampleAverage)
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import (ERPTransformParams, extract_eye_info,
                                  filter_inquiries, get_default_transform)
from matplotlib import pyplot as plt
from scipy import stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

def calculate_stats(train_data: np.ndarray):
    '''
    Calculate the mean and covariance of the training data
    '''
    mu = np.mean(train_data, axis=0)
    sigma = np.cov(train_data.T)
    
    return mu, sigma

def calculate_loglikelihoods(data, mu, sigma) -> np.ndarray:
    # Find the likelihoods by insterting the test data into the pdf of each component
    log_likelihoods = stats.multivariate_normal.logpdf(data, mu, sigma, allow_singular=True)

    return log_likelihoods

def subset_data(data: np.ndarray, labels: np.ndarray, test_size: float, random_state: int = 0, swap_axes: bool = True):
    """Performs a train/test split on the provided data and labels, accounting for
    the current shape convention (channel dimension in front, instead of batch dimension in front).

    Parameters:
    -----------
        data (np.ndarray): Shape (channels, items, time)
        labels (np.ndarray): Shape (items,)
        test_size (float): fraction of data to be used for testing
        random_state (int, optional): fixed random seed
        swap_axes (bool, optional): if true, swaps the axes of the data before splitting

    Returns:
    --------
        train_data (np.ndarray): Shape (channels, train_items, time)
        test_data (np.ndarray): Shape (channels, test_items, time)
        train_labels (np.ndarray): Shape (train_items,)
        test_labels (np.ndarray): Shape (test_items,)
    """
    if swap_axes:
        data = data.swapaxes(0, 1)
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=test_size, random_state=random_state
        )
        train_data = train_data.swapaxes(0, 1)
        test_data = test_data.swapaxes(0, 1)

    else:
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=test_size, random_state=random_state
        )
    return train_data, test_data, train_labels, test_labels


def calculate_acc(predictions: int, counter: int):
    '''
    Compute performance characteristics on the provided test data and labels.

    predictions: predicted labels for each test point per symbol
    counter: true labels for each test point per symbol
    '''
    accuracy_per_symbol = np.sum(predictions == counter) / len(predictions) * 100

    return accuracy_per_symbol


def analyze_multimodal(eeg_data, 
                       gaze_data, 
                       parameters, 
                       device_spec_eeg, 
                       device_spec_gaze, 
                       data_folder, 
                       estimate_balanced_acc,
                       save_figures=False, 
                       show_figures=False):
    """Analyze ERP data and return/save the ERP model.
    Extract relevant information from raw data object.
    Extract timing information from trigger file.
    Apply filtering and preprocessing on the raw data.
    Reshape and label the data for the training procedure.
    Fit the model to the training data. Use cross validation to select parameters.
    Return performance measures on a separate test set.
    Pickle dump model into .pkl file
    Generate and [optional] save/show ERP figures.

    Parameters:
    -----------
        eeg_data (RawData): RawData object containing the data to be analyzed.
        gaze_data (RawData): RawData object containing the data to be analyzed.
        parameters (Parameters): Parameters object retireved from parameters.json.
        device_spec (DeviceSpec): DeviceSpec object containing information about the device used.
        data_folder (str): Path to the folder containing the data to be analyzed.
        estimate_balanced_acc (bool): If true, uses another model copy on an 80/20 split to
            estimate balanced accuracy.
        save_figures (bool): If true, saves ERP figures after training to the data folder.
        show_figures (bool): If true, shows ERP figures after training.
    """

    # Extract relevant session information from parameters file
    trial_window = parameters.get("trial_window")
    if trial_window is None:
        trial_window = (0.0, 0.5)
    window_length = trial_window[1] - trial_window[0]  # eeg window length, in seconds

    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)

    # Get signal filtering information
    transform_params = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate
    static_offset = parameters.get("static_trigger_offset")
    # Get the flash time (for gaze analysis)
    flash_time = parameters.get("time_flash")

    log.info(
        f"\nEEG Data processing settings: \n"
        f"{str(transform_params)} \n"
        f"Trial Window: {trial_window[0]}-{trial_window[1]}s, "
        f"Prestimulus Buffer: {prestim_length}s, Poststimulus Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )
    eeg_channels = eeg_data.channels
    eeg_type_amp = eeg_data.daq_type
    eeg_sample_rate = eeg_data.sample_rate

    gaze_channels = gaze_data.channels
    gaze_type_amp = gaze_data.daq_type
    gaze_sample_rate = gaze_data.sample_rate

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=eeg_sample_rate,
        notch_freq_hz=transform_params.notch_filter_frequency,
        bandpass_low=transform_params.filter_low,
        bandpass_high=transform_params.filter_high,
        bandpass_order=transform_params.filter_order,
        downsample_factor=transform_params.down_sampling_rate,
    )

    log.info(f"Channels read from csv: {eeg_channels}")
    log.info(f"Device type: {eeg_type_amp}, fs={eeg_sample_rate}")
    log.info(f"Channels read from csv: {gaze_channels}")
    log.info(f"Device type: {gaze_type_amp}, fs={gaze_sample_rate}")
    eeg_channel_map = analysis_channels(eeg_channels, device_spec_eeg)
    gaze_channel_map = analysis_channels(gaze_channels, device_spec_gaze)

    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    eeg_channels_used = [eeg_channels[i] for i, keep in enumerate(eeg_channel_map) if keep == 1]
    log.info(f'Channels used in eeg analysis: {eeg_channels_used}')

    gaze_channels_used = [gaze_channels[i] for i, keep in enumerate(gaze_channel_map) if keep == 1]
    log.info(f'Channels used in gaze analysis: {gaze_channels_used}')

    # Define the model object before reshaping the data
    k_folds = parameters.get("k_folds")
    eeg_model = PcaRdaKdeModel(k_folds=k_folds)
    # Select between the two (or three) gaze models to test:
    gaze_model = KernelGPSampleAverage() # change the name to gaussian process model

    # Process triggers.txt files for eeg data:
    trigger_targetness, trigger_timing, inquiry_symbols = trigger_decoder(
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
        offset=static_offset,
        device_type='EEG'
    )

    # Same as above, but with the 'prompt' triggers added for gaze analysis:
    trigger_targetness_gaze, trigger_timing_gaze, trigger_symbols = trigger_decoder(
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        remove_pre_fixation=False,
        exclusion=[
            TriggerType.PREVIEW,
            TriggerType.EVENT,
            TriggerType.FIXATION,
            TriggerType.SYSTEM,
            TriggerType.OFFSET],
        device_type='EYETRACKER',
        apply_starting_offset=False
    )
    ''' Trigger_timing includes PROMPT and excludes FIXATION '''
    
    target_symbols = [trigger_symbols[idx] for idx, targetness in enumerate(trigger_targetness_gaze) if targetness == 'prompt']
    inq_start = trigger_timing_gaze[1::11] # inquiry start times, exluding prompt and fixation

    symbol_set = alphabet()

    # update the trigger timing list to account for the initial trial window
    corrected_trigger_timing = [timing + trial_window[0] for timing in trigger_timing]

    erp_data, fs_eeg = eeg_data.by_channel()
    trajectory_data, fs_eye = gaze_data.by_channel() 

    # Reshaping EEG data:
    eeg_inquiries, eeg_inquiry_labels, eeg_inquiry_timing = eeg_model.reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=corrected_trigger_timing,
        eeg_data=erp_data,
        sample_rate=eeg_sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=eeg_channel_map,
        poststimulus_length=window_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,
    )
    # Size = Inquiries x Channels x Samples

    # Reshaping gaze data:
    gaze_inquiries_dict, gaze_inquiries_list, _ = gaze_model.reshaper(
        inq_start_times=inq_start,
        target_symbols=target_symbols,
        gaze_data=trajectory_data,
        sample_rate=gaze_sample_rate,
        stimulus_duration=flash_time,
        num_stimuli_per_inquiry=10,
        symbol_set=symbol_set
    )

    ## More EEG preprocessing:
    eeg_inquiries, fs = filter_inquiries(eeg_inquiries, default_transform, eeg_sample_rate)
    eeg_inquiry_timing = update_inquiry_timing(eeg_inquiry_timing, downsample_rate)
    trial_duration_samples = int(window_length * fs)
    # preprocessed_eeg_data = eeg_model.reshaper.extract_trials(eeg_inquiries, trial_duration_samples, eeg_inquiry_timing)
    # we will extract the trials from the inquiries later

    # define the training classes using integers, where 0=nontargets/1=targets
    # erp_labels = eeg_inquiry_labels.flatten().tolist()

    
    ## More gaze preprocessing:
    preprocessed_gaze_data = np.zeros((len(gaze_inquiries_list), 4, 180))
    # Extract left_x, left_y, right_x, right_y for each inquiry
    for j in range(len(gaze_inquiries_list)):
        left_eye, right_eye, _, _, _, _ = extract_eye_info(gaze_inquiries_list[j])
        preprocessed_gaze_data[j] = np.concatenate((left_eye.T, right_eye.T,), axis=0)

    preprocessed_gaze_dict = {i: [] for i in symbol_set}
    for i in symbol_set:
        # Skip if there's no evidence for this symbol:
        if len(gaze_inquiries_dict[i]) == 0:
            continue
        for j in range(len(gaze_inquiries_dict[i])):
            left_eye, right_eye, _, _, _, _ = extract_eye_info(gaze_inquiries_dict[i][j])
            preprocessed_gaze_dict[i].append((np.concatenate((left_eye.T, right_eye.T), axis=0)))   
        preprocessed_gaze_dict[i] = np.array(preprocessed_gaze_dict[i])
    
    # Find the time averages for each symbol:
    centralized_data_dict = {i: [] for i in symbol_set}
    time_average_per_symbol = {i: [] for i in symbol_set}
    for sym in symbol_set:
    # Skip if there's no evidence for this symbol:
        try: 
            if len(gaze_inquiries_dict[sym]) == 0:
                continue
        except:
            log.info(f"Error in symbol {sym}")
            continue
        
        # with open(f"{BCIPY_ROOT}/parameters/symbol_positions.json", 'r') as params_file:
        #         symbol_positions = json.load(params_file)

        for j in range(len(preprocessed_gaze_dict[sym])):
            temp = np.mean(preprocessed_gaze_dict[sym][j], axis=1)
            time_average_per_symbol[sym].append(temp)
            centralized_data_dict[sym].append(gaze_model.substract_mean(preprocessed_gaze_dict[sym][j], temp))  # Delta_t = X_t - mu
        centralized_data_dict[sym] = np.array(centralized_data_dict[sym])
        time_average_per_symbol[sym] = np.mean(np.array(time_average_per_symbol[sym]), axis=0)
    # print(f"time_average for symbol {sym}: ", time_average[sym])

    # Take the time average of the gaze data:
    centralized_gaze_data = np.zeros_like(preprocessed_gaze_data)
    for i, (_, sym) in enumerate(zip(preprocessed_gaze_data, target_symbols)):
        centralized_gaze_data[i] = gaze_model.substract_mean(preprocessed_gaze_data[i], time_average_per_symbol[sym])

    '''Use bootstrap resampling for both EEG and Gaze data'''
    log.info("Bootstrap sampling for EEG and Gaze data...")
    n_iterations=1
    for i in range(n_iterations):
        # Pick a train and test dataset (that consists of non-train elements) until test dataset is not empty:
        train_indices = resample(list(range(100)), replace=True, n_samples=150, random_state=0)
        test_indices = np.array([x for x in list(range(100)) if x not in train_indices])
        if len(test_indices) == 0:
            break

        train_data_eeg = eeg_inquiries[:, train_indices, :]
        test_data_eeg = eeg_inquiries[:, test_indices, :]
        eeg_inquiry_timing = np.array(eeg_inquiry_timing)
        train_eeg_inquiry_timing = eeg_inquiry_timing[train_indices]
        test_eeg_inquiry_timing = eeg_inquiry_timing[test_indices]


        # Now extract the inquiries from trials for eeg model fitting:
        preprocessed_train_eeg = eeg_model.reshaper.extract_trials(train_data_eeg, trial_duration_samples, train_eeg_inquiry_timing)
        preprocessed_test_eeg = eeg_model.reshaper.extract_trials(test_data_eeg, trial_duration_samples, test_eeg_inquiry_timing)

        # train and save the eeg model a pkl file
        log.info("Training model. This will take some time...")
        # Flatten the labels (0=nontarget/1=target) prior to model fitting
        erp_train_labels = eeg_inquiry_labels[train_indices].flatten().tolist()
        erp_test_labels = eeg_inquiry_labels[test_indices].flatten().tolist()
        eeg_model.fit(preprocessed_train_eeg, erp_train_labels)
        eeg_model.metadata = SignalModelMetadata(device_spec=device_spec_eeg,
                                            transform=default_transform)
        log.info(f"Training complete [AUC={eeg_model.auc:0.4f}]. Saving data...")
        # save_model(eeg_model, Path(data_folder, f"model_{eeg_model.auc:0.4f}.pkl"))
        preferences.signal_model_directory = data_folder

        # Given the test data, compute the log likelihood ratios for each symbol:
        eeg_likelihood_ratios = eeg_model.compute_likelihood_ratio(preprocessed_test_eeg, inquiry_symbols, alphabet())
        eeg_log_likelihood_ratios = np.log(eeg_likelihood_ratios)
        breakpoint()

        # train and save the gaze model as a pkl file

        # And the log likelihood evidence from gaze model:


        # Bayesian fusion update and decision making:

    
    '''
    # Using an 80/20 split, report on balanced accuracy
    
    if estimate_balanced_acc:
        # Implement cross-validation for balanced accuracy
        scores = []

        skf = KFold(n_splits=5, shuffle=True, random_state=0)
        data = data.swapaxes(0, 1)
        for train_index, test_index in skf.split(data, labels):
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = np.array(labels)[train_index], np.array(labels)[test_index]
            dummy_model = PcaRdaKdeModel(k_folds=k_folds)
            breakpoint()
            # Swap axes back to original shape:
            train_data = train_data.swapaxes(0, 1)
            test_data = test_data.swapaxes(0, 1)
            dummy_model.fit(train_data, train_labels)
            probs = dummy_model.predict_proba(test_data)
            preds = probs.argmax(-1)
            breakpoint()
            score = balanced_accuracy_score(test_labels, preds)
            scores.append(score)

        average_score = np.mean(scores)
        log.info(f"Cross-validated balanced accuracy: {average_score}")

        del dummy_model, train_data, test_data, train_labels, test_labels, probs, preds
        '''

    # this should have uncorrected trigger timing for display purposes
    figure_handles = visualize_erp(
        eeg_data,
        eeg_channel_map,
        trigger_timing,
        erp_labels,
        trial_window,
        transform=default_transform,
        plot_average=True,
        plot_topomaps=True,
        save_path=data_folder if save_figures else None,
        show=show_figures
    )
    return eeg_model, figure_handles


def analyze_gaze(
        gaze_data,
        parameters,
        device_spec,
        data_folder,
        save_figures=None,
        show_figures=False,
        plot_points=False,
        model_type="Individual"):
    """Analyze gaze data and return/save the gaze model.
    Extract relevant information from gaze data object.
    Extract timing information from trigger file.
    Apply preprocessing on the raw data. Extract the data for each target label and each eye separately.
    Extract inquiries dictionary with keys as target symbols and values as inquiry windows.
    Based on the model type, fit the model to the data.
    Pickle dump model into .pkl file
    Generate and [optional] save/show gaze figures.

    Parameters:
    -----------
        gaze_data (RawData): RawData object containing the data to be analyzed.
        parameters (Parameters): Parameters object retireved from parameters.json.
        device_spec (DeviceSpec): DeviceSpec object containing information about the device used.
        data_folder (str): Path to the folder containing the data to be analyzed.
        save_figures (bool): If true, saves ERP figures after training to the data folder.
        show_figures (bool): If true, shows ERP figures after training.
        plot_points (bool): If true, plots the gaze points on the matrix image.
        model_type (str): Type of gaze model to be used. Options are:
            "Individual": Fits a separate Gaussian for each symbol. Default model
            "Centralized": Uses data from all symbols to fit a single centralized Gaussian
    """
    figures = []
    figure_handles = visualize_gaze(
        gaze_data,
        save_path=data_folder if save_figures else None,
        img_path=f'{data_folder}/{MATRIX_IMAGE_FILENAME}',
        show=show_figures,
        raw_plot=True,
    )
    figures.append(figure_handles)

    channels = gaze_data.channels
    type_amp = gaze_data.daq_type
    sample_rate = gaze_data.sample_rate

    flash_time = parameters.get("time_flash")  # duration of each stimulus
    stim_size = parameters.get("stim_length")  # number of stimuli per inquiry

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {type_amp}, fs={sample_rate}")
    channel_map = analysis_channels(channels, device_spec)

    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    log.info(f'Channels used in analysis: {channels_used}')

    data, _fs = gaze_data.by_channel()

    if model_type == "GP_SampleAverage":
        model = KernelGPSampleAverage()
    # elif model_type == "Centralized":
    #     model = GazeModelCombined()
    # elif model_type == "GP":
    #     model = GazeModelKernelGaussianProcess()
    # elif model_type == "GP_SampleAverage":
    #     model = GazeModelKernelGPSampleAverage()

    # # Extract all Triggers info
    # trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
    #     trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
    #     remove_pre_fixation=True,
    #     exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    #     device_type='EYETRACKER',
    #     apply_starting_offset=False
    # )

    # Extract all Triggers info
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        remove_pre_fixation=False,
        exclusion=[
            TriggerType.PREVIEW,
            TriggerType.EVENT,
            TriggerType.FIXATION,
            TriggerType.SYSTEM,
            TriggerType.OFFSET],
        device_type='EYETRACKER',
        apply_starting_offset=False
    )
    ''' Trigger_timing includes PROMPT and excludes FIXATION '''

    # Extract the inquiries dictionary with keys as target symbols and values as inquiry windows:
    symbol_set = alphabet()

    target_symbols = trigger_symbols[0::11]  # target symbols are the PROMPT triggers
    # Use trigger_timing to generate time windows for each letter flashing
    # Take every 10th trigger as the start point of timing.
    inq_start = trigger_timing[1::11]  # start of each inquiry (here we jump over prompts)

    # Extract the inquiries dictionary with keys as target symbols and values as inquiry windows:
    inquiries_dict, inquiries_list, labels = model.reshaper(
        inq_start_times=inq_start,
        target_symbols=target_symbols,
        gaze_data=data,
        sample_rate=sample_rate,
        stimulus_duration=flash_time,
        num_stimuli_per_inquiry=10,
        symbol_set=symbol_set
    )

    # Extract the data for each target label and each eye separately.
    # Apply preprocessing:
    preprocessed_data = {i: [] for i in symbol_set}
    for i in symbol_set:
        # Skip if there's no evidence for this symbol:
        if len(inquiries_dict[i]) == 0:
            continue

        for j in range(len(inquiries_dict[i])):
            left_eye, right_eye, left_pupil, right_pupil, deleted_samples, all_samples = extract_eye_info(inquiries_dict[i][j])
            preprocessed_data[i].append((np.concatenate((left_eye.T, right_eye.T), axis=0)))   
             # Inquiries x All Dimensions (left_x, left_y, right_x, right_y) x Time
        preprocessed_data[i] = np.array(preprocessed_data[i])
    
    centralized_data_left = []
    centralized_data_right = []
    centralized_data_train = {i: [] for i in symbol_set}
    centralized_data = {i: [] for i in symbol_set}
    time_average = {i: [] for i in symbol_set}
    train_dict = {}
    test_dict = {i: [] for i in symbol_set}

    left_eye_all = []
    right_eye_all = []

    for sym in symbol_set:
        # Skip if there's no evidence for this symbol:
        try: 
            if len(inquiries_dict[sym]) == 0:
                continue
            # le = preprocessed_data[sym][0]
            # re = preprocessed_data[sym][1]
            # breakpoint()

        #     # Train test split:
        #     labels = np.array([sym] * len(le))  # Labels are the same for both eyes
        #     train_le, test_le, train_labels_le, test_labels_le = subset_data(le, labels, test_size=0.2, swap_axes=False)
        #     train_re, test_re, train_labels_re, test_labels_re = subset_data(re, labels, test_size=0.2, swap_axes=False)
        #     train_dict[sym] = np.concatenate((train_le, train_re), axis=0)  # Note that here both eyes are concatenated
        #     test_dict[sym] = np.concatenate((test_le, test_re), axis=0)

        except:
            log.info(f"Error in symbol {sym}")
            continue

        # Fit the model based on model type.
        if model_type == "Individual":
            # Model 1: Fit Gaussian mixture on each symbol separately
            model.fit(train_dict[sym])


            left_eye_all.append(le)
            right_eye_all.append(re)


        # if model_type == "Centralized":
        if model_type == "GP_SampleAverage":
            # Centralize the data using symbol positions:
            # Load json file.
            with open(f"{BCIPY_ROOT}/parameters/symbol_positions.json", 'r') as params_file:
                symbol_positions = json.load(params_file)
            # Subtract the symbol positions from the data:
            # centralized_data_left.append(model.centralize(train_le, symbol_positions[sym]))
            # centralized_data_right.append(model.centralize(train_re, symbol_positions[sym]))
            # centralized_data_train.append(model.centralize(train_dict[sym], symbol_positions[sym]))

            # for j in range(len(preprocessed_data[sym])):
            #     centralized_data[sym].append(model.centralize(preprocessed_data[sym][j], symbol_positions[sym]))

            # Instead of centralizing, take the time average:
            for j in range(len(preprocessed_data[sym])):
                temp = np.mean(preprocessed_data[sym][j], axis=1)
                time_average[sym].append(temp)
                centralized_data[sym].append(model.substract_mean(preprocessed_data[sym][j], temp))  # Delta_t = X_t - mu
            centralized_data[sym] = np.array(centralized_data[sym])
            time_average[sym] = np.mean(np.array(time_average[sym]), axis=0)
            # print(f"time_average for symbol {sym}: ", time_average[sym])


    if model_type == "Individual":
        accuracy = 0
        acc_all_symbols = {}
        counter = 0

        for sym in symbol_set:
            # Continue if there is no test data for this symbol:
            if len(test_dict[sym]) == 0:
                acc_all_symbols[sym] = 0
                continue
            # TODO: likelihoods should be in predict_proba !!!
            predictions = model.predict(  # inference!
                test_dict[sym])
            acc_all_symbols[sym] = calculate_acc(predictions, counter) # TODO use evaluate method
            accuracy += acc_all_symbols[sym]
            counter += 1
        accuracy /= counter

        # df = pd.DataFrame(acc_all_symbols, index=[0])
        # df.to_csv('my_data.csv', index=False)

        # Plot all accuracies as bar plot:
        figure_handles = visualize_gaze_accuracies(acc_all_symbols, 
                                                   accuracy, 
                                                   save_path=data_folder if save_figures else None, 
                                                   show=show_figures)
        figures.append(figure_handles)

    if model_type == "GP_SampleAverage":
        # Visualize different inquiries from the same target letter:
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w', 'orange', 'purple']
        for sym in symbol_set:
            if len(centralized_data[sym]) == 0:
                continue
            # for j in range(len(centralized_data[sym])):
            #     plt.plot(range(len(centralized_data[sym][j][:, 0])), centralized_data[sym][j][:, 0], label=f'{sym} Inquiry {j}', c=colors[j])
            # plt.legend()
            # plt.show()
            # breakpoint()
       
        # # Save the dictionary as a csv file:
        # df = pd.DataFrame.from_dict(centralized_data, orient='index')
        # df.to_csv('centralized_data.csv', header=False)

        # # Also save preprocessed data:
        # df_2 = pd.DataFrame.from_dict(preprocessed_data, orient='index')
        # df_2.to_csv('preprocessed_data.csv', header=False)

        # Split the data into train and test sets & fit the model:
        centralized_data_training_set = []
        for sym in symbol_set:
            if len(centralized_data[sym]) <= 1:
                if len(centralized_data[sym]) == 1:
                    test_dict[sym] = preprocessed_data[sym][-1]
                continue
            # Leave one out and add the rest to the training set:
            for j in range(len(centralized_data[sym])-1):
                centralized_data_training_set.append(centralized_data[sym][j])
            # Add the last inquiry to the test set:
            test_dict[sym] = preprocessed_data[sym][-1]

        # Save the test dict as well:
        import pickle 
        with open('test_dict.pkl', 'wb') as f:
            pickle.dump(test_dict, f)
        
        # Save the list as well:
        with open('centralized_data_training_set.pkl', 'wb') as f:
            pickle.dump(centralized_data_training_set, f)

        centralized_data_training_set = np.array(centralized_data_training_set)

        # Take the sample average of the centralized data: time_average
        # centralized_data_training_set.shape = (72,4,180)
        
        # flatten the covariance to (72, 720)
        # cov_matrix = np.zeros((centralized_data_training_set.shape[0], centralized_data_training_set.shape[1]*centralized_data_training_set.shape[2]))
        
        # for i in range(centralized_data_training_set.shape[0]):
        reshaped_data = centralized_data_training_set.reshape((72,720))
        cov_matrix = np.cov(reshaped_data, rowvar=False)
        # cov_matrix.shape = (720,720)
        plt.imshow(cov_matrix)
        plt.colorbar()
        plt.show()
        reshaped_mean = np.mean(reshaped_data, axis=0)

        # Find the likelihoods for the test case:
        l_likelihoods = np.zeros((len(symbol_set), len(symbol_set)))
        log_likelihoods = np.zeros((len(symbol_set), len(symbol_set)))
        counter = 0
        for i_sym0, sym0 in enumerate(symbol_set):
            for i_sym1, sym1 in enumerate(symbol_set):
                if len(centralized_data[sym1]) == 0:
                    continue
                if len(test_dict[sym0]) == 0:
                    continue
                # print(f"Target: {sym0}, Tested: {sym1}")
                central_data = model.substract_mean(test_dict[sym0], time_average[sym1])
                flattened_data = central_data.reshape((720,))
                diff = flattened_data - reshaped_mean
                eps = 1e-6
                inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(len(cov_matrix))*eps)
                numerator = -np.dot(diff.T, np.dot(inv_cov_matrix, diff))/2
                # denominator = np.log(np.linalg.det(cov_matrix + np.eye(len(cov_matrix))*eps))/2+np.log(2*np.pi)*len(cov_matrix)/2
                #TODO: Fix -inf error  
                denominator = 0
                log_likelihood = numerator - denominator
                # print(f"{log_likelihood:.3E}")
                log_likelihoods[i_sym0, i_sym1] = log_likelihood
            # Find the max likelihood:
            max_like = np.argmax(log_likelihoods[i_sym0, :])
            # Check if it's the same as the target, and save the result:
            if max_like == i_sym0:
                # print("True")
                counter += 1

                # l_likelihoods[i_sym0, i_sym1] = calculate_loglikelihoods(flattened_data, reshaped_mean,  cov_matrix)
        # print(central_data)
        # print("log_likelihoods: ")
        # print(log_likelihoods)
        breakpoint()
        # max_like = l_likelihoodsl_likelihoods.max(axis=0)

        # Find the covariances of the centralized data:
        cov_matrix_trial = np.zeros((centralized_data_training_set.shape[1], centralized_data_training_set.shape[0], centralized_data_training_set.shape[0]))
        for i_coord in range(centralized_data_training_set.shape[1]):
            cov_matrix_trial[i_coord] = np.cov(centralized_data_training_set[:, i_coord, :])
        # cov_matrix_trial.shape = (4,72,72)
        plt.imshow(cov_matrix_trial[0])
        plt.show()
        
        cov_matrix_coord = np.zeros((centralized_data_training_set.shape[2], centralized_data_training_set.shape[1], centralized_data_training_set.shape[1]))
        for i_time in range(centralized_data_training_set.shape[2]):
            cov_matrix_coord[i_time] = np.cov((centralized_data_training_set[:, :, i_time]).T)
        # cov_matrix_coord.shape = (180,4,4)
        plt.imshow(cov_matrix_coord[0])
        plt.show()

        cov_matrix_time = np.zeros((centralized_data_training_set.shape[1], centralized_data_training_set.shape[2], centralized_data_training_set.shape[2]))
        for i_coord in range(centralized_data_training_set.shape[1]):
            cov_matrix_time[i_coord] = np.cov((centralized_data_training_set[:, i_coord, :]).T)
        # cov_matrix_time.shape = (4,180,180)    -> Chosen Kernel
        # Plot the correlation between the time points
        corr = np.corrcoef(centralized_data_training_set[:, 0, :].T) # left_x over all time points
        # plt.imshow(corr)
        fig, ax = plt.subplots()
        cax = ax.matshow(corr, cmap='coolwarm')
        fig.colorbar(cax)
        plt.show()

        # Plot the correlation between the different dimensions, with colorbar:
        # corr = np.corrcoef(centralized_data_training_set[0, :, :])

        # Find the mean of the centralized_data_training_set 
        mean_delta = np.mean(centralized_data_training_set, axis=0)
        dim_list = ['Left_x', 'Left_y', 'Right_x', 'Right_y']
        fig, axs = plt.subplots(4,1)
        for i, dim in zip(range(mean_delta.shape[0]), dim_list):    
            axs[i].plot(range(len(mean_delta[i, :])), mean_delta[i, :], label=f'Mean Inquiry {dim}', c=colors[i])
            std_dev = np.sqrt(np.diag(cov_matrix_time[i]))
            axs[i].fill_between(range(len(mean_delta[i, :])), mean_delta[i, :]- std_dev, mean_delta[i, :]+std_dev, alpha=0.2)
            axs[i].legend()
            axs[i].set_ylim(-0.01, 0.01)
        plt.suptitle('Sample Average & Confidence Interval for Inquiries')

        # plt.plot(range(len(mean_delta[0, :])), mean_delta[0, :], label=f'Mean Inquiry Left_x', c='r')
        # plt.fill_between(range(len(mean_delta[0, :])), mean_delta[0, :]- std_for_left_x, mean_delta[0, :]+st_dev_for_left_x)
        plt.show() 

        # for j in range(len(centralized_data_training_set)):
        #     cov_matrix =+ np.cov(centralized_data_training_set[j,:,:], rowvar=False)
        # cov_matrix = cov_matrix / len(centralized_data_training_set)
        # cov_matrix.shape = (180,180)

        # for sym in symbol_set:
        #     for j in range(len(centralized_data[sym])):
        #         plt.plot(range(len(centralized_data[sym][j][0, :])), centralized_data[sym][j][0, :], label=f'{sym} Inquiry {j} Left_x', c=colors[j])
        #     plt.legend()
        #     plt.show()
        
        breakpoint()
        
        # model.fit(centralized_data_training_set)

    if model_type == "Centralized":
        # Model 2: Fit Gaussian mixture on a centralized data

        all_data = np.concatenate(centralized_data_train, axis=0)
    
        # # Visualize the results:
        # figure_handles = visualize_centralized_data(
        #     all_data,
        #     save_path=data_folder if save_figures else None,
        #     show=show_figures,
        #     img_path=f"{data_folder}/matrix.png",
        #     raw_plot=True,
        # )
        # figures.append(figure_handles)

        # Calculate means, covariances for each symbol.
        for sym in symbol_set:
            if len(test_dict[sym]) == 0:
                continue

            # Visualize the results:
            le = preprocessed_data[sym][0]
            re = preprocessed_data[sym][1]
            figure_handles = visualize_gaze_inquiries(
                le, re,
                model.means, model.covs,
                save_path=None,
                show=False,
                img_path=f"{data_folder}/matrix.png",
                raw_plot=True,
            )
            figures.append(figure_handles)
            left_eye_all.append(le)
            right_eye_all.append(re)

        # Compute scores for the test set.
        accuracy = 0
        acc_all_symbols = {}
        counter = 0
        for sym in symbol_set:
            if len(test_dict[sym]) == 0:
                # Continue if there is no test data for this symbol:
                acc_all_symbols[sym] = 0
                continue
            predictions = model.predict(
                test_dict[sym])
            acc_all_symbols[sym] = calculate_acc(predictions, counter) # TODO use evaluate method
            accuracy += acc_all_symbols[sym]
            counter += 1
        accuracy /= counter

        # Plot all accuracies as bar plot:
        figure_handles = visualize_gaze_accuracies(acc_all_symbols, 
                                                   accuracy, 
                                                   save_path=data_folder if save_figures else None, 
                                                   show=show_figures)
        
        # Save accuracies for later use as a csv file:
        # df = pd.DataFrame(acc_all_symbols, index=[0])
        # df.to_csv('my_data.csv', index=False)
        # basak = pd.read_csv('my_data.csv')
        
        figures.append(figure_handles)

    fig_handles = visualize_results_all_symbols(
        left_eye_all, right_eye_all,
        model.means, model.covs,
        save_path=data_folder if save_figures else None,
        show=show_figures,
        img_path=f"{data_folder}/matrix.png",
        raw_plot=True,
    )
    figures.append(fig_handles)

    model.metadata = SignalModelMetadata(device_spec=device_spec,
                                         transform=None)
    log.info("Training complete for Eyetracker model. Saving data...")
    save_model(
        model,
        Path(data_folder, f"model_{device_spec.content_type}_{model_type}.pkl"))
    return model, figures


@report_execution_time
def offline_analysis(
    data_folder: str = None,
    parameters: Parameters = None,
    alert_finished: bool = True,
    estimate_balanced_acc: bool = True,
    show_figures: bool = False,
    save_figures: bool = False,
) -> Tuple[SignalModel, Figure]:
    """Gets calibration data and trains the model in an offline fashion.
    pickle dumps the model into a .pkl folder

    How it Works:
    For every active device that was used during calibration,
    - reads data and information from a .csv calibration file
    - reads trigger information from a .txt trigger file
    - filters data
    - reshapes and labels the data for the training procedure
    - fits the model to the data
        - uses cross validation to select parameters
        - based on the parameters, trains system using all the data
    - pickle dumps model into .pkl file
    - generates and [optional] saves/shows the ERP figure
    - [optional] alert the user finished processing

    For gaze modeling, user can set one of the following 4 models with corresponding keywords:
    Individual: Gaze model that fits different Gaussians/Gaussian Mixtures for each symbol.
    Centralized: Gaze model that uses all symbols to fit a single Gaussian, by centralizing the input
    GP: Multi-Output Gaussian Process 
    GP_SampleAverage: Multi-Output Gaussian Process with centralized input.

    Parameters:
    ----------
        data_folder(str): folder of the data
            save all information and load all from this folder
        parameter(dict): parameters for running offline analysis
        alert_finished(bool): whether or not to alert the user offline analysis complete
        estimate_balanced_acc(bool): if true, uses another model copy on an 80/20 split to
            estimate balanced accuracy
        show_figures(bool): if true, shows ERP figures after training
        save_figures(bool): if true, saves ERP figures after training to the data folder

    Returns:
    --------
        model (SignalModel): trained model
        figure_handles (Figure): handles to the ERP figures
    """
    assert parameters, "Parameters are required for offline analysis."
    if not data_folder:
        data_folder = load_experimental_data()

    devices_by_name = devices.load(
        Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME), replace=True)

    active_devices = (spec for spec in devices_by_name.values()
                      if spec.is_active)
    active_raw_data_paths = (Path(data_folder, raw_data_filename(device_spec))
                             for device_spec in active_devices)
    data_file_paths = [path for path in active_raw_data_paths if path.exists()]

    eeg_data = load_raw_data(data_file_paths[0])
    device_spec_eeg = devices_by_name.get(eeg_data.daq_type)
    gaze_data = load_raw_data(data_file_paths[1])
    device_spec_gaze = devices_by_name.get(gaze_data.daq_type)

      
    # Analyze both EEG and Eyetracker data here
    multi_model, multi_figure_handles = analyze_multimodal(
                eeg_data, gaze_data, parameters, device_spec_eeg, device_spec_gaze, data_folder, 
                  estimate_balanced_acc, save_figures, show_figures)
            # models.append(erp_model)
            # figure_handles.append(erp_figure_handles)

    # et_model, et_figure_handles = analyze_gaze(
    #             gaze_data, parameters, device_spec_gaze, data_folder, save_figures, 
    #             show_figures, model_type="GP_SampleAverage")
            # models.append(et_model)
            # figure_handles.append(et_figure_handles)

    if alert_finished:
        play_sound(f"{STATIC_AUDIO_PATH}/{parameters['alert_sound_file']}")
    # return multi_model, multi_figure_handles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    parser.add_argument("-p", "--parameters_file", default=DEFAULT_PARAMETERS_PATH)
    parser.add_argument("-s", "--save_figures", action="store_true")
    parser.add_argument("-v", "--show_figures", action="store_true")
    parser.add_argument("--alert", dest="alert", action="store_true")
    parser.add_argument("--balanced-acc", dest="balanced", action="store_true")
    parser.set_defaults(alert=False)
    parser.set_defaults(balanced=True)
    parser.set_defaults(save_figures=True)
    parser.set_defaults(show_figures=True)
    args = parser.parse_args()

    log.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)

    offline_analysis(
        args.data_folder,
        parameters,
        alert_finished=args.alert,
        estimate_balanced_acc=args.balanced,
        save_figures=args.save_figures,
        show_figures=args.show_figures)
    log.info("Offline Analysis complete.")
