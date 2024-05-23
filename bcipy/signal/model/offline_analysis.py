# mypy: disable-error-code="attr-defined"
# needed for the ERPTransformParams
import logging
from pathlib import Path
from typing import Tuple
import json

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

import bcipy.acquisition.devices as devices
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME, BCIPY_ROOT,
                          DEFAULT_PARAMETERS_PATH, STATIC_AUDIO_PATH,
                          TRIGGER_FILENAME)
from bcipy.helpers.acquisition import analysis_channels, raw_data_filename
from bcipy.helpers.load import (load_experimental_data, load_json_parameters,
                                load_raw_data)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import save_model
from bcipy.helpers.stimuli import play_sound, update_inquiry_timing
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.visualization import (visualize_erp, visualize_gaze,
                                         visualize_gaze_inquiries,
                                         visualize_centralized_data,
                                         visualize_results_all_symbols,
                                         visualize_gaze_accuracies)
from bcipy.preferences import preferences
from bcipy.signal.model.base_model import SignalModel, SignalModelMetadata
from bcipy.signal.model.gaussian_mixture import GazeModelIndividual, GazeModelCombined
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import (ERPTransformParams, extract_eye_info,
                                  filter_inquiries, get_default_transform)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")


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


def analyze_erp(erp_data, parameters, device_spec, data_folder, estimate_balanced_acc,
                save_figures=False, show_figures=False):
    """Analyze ERP data and return/save the ERP model.
    Extract relevant information from raw data object.
    Extract timing information from trigger file.
    Apply filtering and preprocessing on the raw data.
    Reshape and label the data for the training procedure.
    Fit the model to the data. Use cross validation to select parameters.
    Pickle dump model into .pkl file
    Generate and [optional] save/show ERP figures.

    Parameters:
    -----------
        erp_data (RawData): RawData object containing the data to be analyzed.
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
    window_length = trial_window[1] - trial_window[0]

    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)

    # Get signal filtering information
    transform_params = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate
    static_offset = parameters.get("static_trigger_offset")

    log.info(
        f"\nData processing settings: \n"
        f"{str(transform_params)} \n"
        f"Trial Window: {trial_window[0]}-{trial_window[1]}s, "
        f"Prestimulus Buffer: {prestim_length}s, Poststimulus Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )
    channels = erp_data.channels
    type_amp = erp_data.daq_type
    sample_rate = erp_data.sample_rate

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=transform_params.notch_filter_frequency,
        bandpass_low=transform_params.filter_low,
        bandpass_high=transform_params.filter_high,
        bandpass_order=transform_params.filter_order,
        downsample_factor=transform_params.down_sampling_rate,
    )

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {type_amp}, fs={sample_rate}")

    k_folds = parameters.get("k_folds")
    model = PcaRdaKdeModel(k_folds=k_folds)

    # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
        offset=static_offset,
        device_type='EEG'
    )

    # update the trigger timing list to account for the initial trial window
    corrected_trigger_timing = [timing + trial_window[0] for timing in trigger_timing]

    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
    log.info(f'Channels used in analysis: {channels_used}')

    data, fs = erp_data.by_channel()

    inquiries, inquiry_labels, inquiry_timing = model.reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=corrected_trigger_timing,
        eeg_data=data,
        sample_rate=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=window_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,
    )

    inquiries, fs = filter_inquiries(inquiries, default_transform, sample_rate)
    inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
    trial_duration_samples = int(window_length * fs)
    data = model.reshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

    # define the training classes using integers, where 0=nontargets/1=targets
    labels = inquiry_labels.flatten().tolist()

    # train and save the model as a pkl file
    log.info("Training model. This will take some time...")
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.fit(data, labels)
    model.metadata = SignalModelMetadata(device_spec=device_spec,
                                         transform=default_transform)
    log.info(f"Training complete [AUC={model.auc:0.4f}]. Saving data...")

    save_model(model, Path(data_folder, f"model_{model.auc:0.4f}.pkl"))
    preferences.signal_model_directory = data_folder

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
            dummy_model.fit(train_data, train_labels)
            probs = dummy_model.predict_proba(test_data)
            preds = probs.argmax(-1)
            score = balanced_accuracy_score(test_labels, preds)
            scores.append(score)
        
        # for i in range(5):
        #     train_data, test_data, train_labels, test_labels = subset_data(data, labels, test_size=0.2, swap_axes=True, random_state=i)
        #     dummy_model = PcaRdaKdeModel(k_folds=k_folds)
        #     dummy_model.fit(train_data, train_labels)
        #     probs = dummy_model.predict_proba(test_data)
        #     preds = probs.argmax(-1)
        #     score = balanced_accuracy_score(test_labels, preds)
        #     scores.append(score)

        average_score = np.mean(scores)
        log.info(f"Cross-validated balanced accuracy: {average_score}")

        del dummy_model, train_data, test_data, train_labels, test_labels, probs, preds

    # this should have uncorrected trigger timing for display purposes
    figure_handles = visualize_erp(
        erp_data,
        channel_map,
        trigger_timing,
        labels,
        trial_window,
        transform=default_transform,
        plot_average=True,
        plot_topomaps=True,
        save_path=data_folder if save_figures else None,
        show=show_figures
    )
    return model, figure_handles


def analyze_gaze(
        gaze_data,
        parameters,
        device_spec,
        data_folder,
        save_figures=False,
        show_figures=False,
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
        model_type (str): Type of gaze model to be used. Options are:
            "Individual": Fits a separate Gaussian for each symbol. Default model
            "Centralized": Uses data from all symbols to fit a single centralized Gaussian
    """
    figures = []
    figure_handles = visualize_gaze(
        gaze_data,
        save_path=data_folder if save_figures else None,
        show=show_figures,
        img_path=f"{data_folder}/matrix.png",
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

    if model_type == "Individual":
        model = GazeModelIndividual()
    elif model_type == "Centralized":
        model = GazeModelCombined()

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

    target_symbols = trigger_symbols[0::11]  # target symbols are the PROMPT triggers
    # Use trigger_timing to generate time windows for each letter flashing
    # Take every 10th trigger as the start point of timing.
    inq_start = trigger_timing[1::11]  # start of each inquiry (here we jump over prompts)

    # Extract the inquiries dictionary with keys as target symbols and values as inquiry windows:
    symbol_set = alphabet()
    inquiries = model.reshaper(
        inq_start_times=inq_start,
        target_symbols=target_symbols,
        gaze_data=data,
        sample_rate=sample_rate,
        stimulus_duration=flash_time,
        num_stimuli_per_inquiry=stim_size,
        symbol_set=symbol_set
    )

    # Extract the data for each target label and each eye separately.
    # Apply preprocessing:
    preprocessed_data = {i: [] for i in symbol_set}
    for i in symbol_set:
        # Skip if there's no evidence for this symbol:
        if len(inquiries[i]) == 0:
            continue

        left_eye, right_eye, left_pupil, right_pupil, deleted_samples, all_samples = extract_eye_info(inquiries[i])
        preprocessed_data[i] = np.array([left_eye, right_eye])    # Channels x Sample Size x Dimensions(x,y)

    centralized_data_train = []
    train_dict = {}
    test_dict = {}

    left_eye_all = []
    right_eye_all = []
    means_all = []
    covs_all = []
    for sym in symbol_set:
        # Skip if there's no evidence for this symbol:
        try: 
            if len(inquiries[sym]) == 0:
                test_dict[sym] = []
                continue
            le = preprocessed_data[sym][0]
            re = preprocessed_data[sym][1]
            # breakpoint()

            # Train test split:
            labels = np.array([sym] * len(le))  # Labels are the same for both eyes
            train_le, test_le, train_labels_le, test_labels_le = subset_data(le, labels, test_size=0.2, swap_axes=False)
            train_re, test_re, train_labels_re, test_labels_re = subset_data(re, labels, test_size=0.2, swap_axes=False)
            train_dict[sym] = np.concatenate((train_le, train_re), axis=0)
            test_dict[sym] = np.concatenate((test_le, test_re), axis=0)

        except:
            log.info(f"Error in symbol {sym}")
            continue

        # Fit the model based on model type.
        if model_type == "Individual":
            # Model 1: Fit Gaussian mixture on each symbol separately
            model.fit(train_dict[sym])

            means, covs = model.evaluate()

            left_eye_all.append(le)
            right_eye_all.append(re)
            means_all.append(means)
            covs_all.append(covs)

        if model_type == "Centralized":
            # Centralize the data using symbol positions:
            # Load json file.
            with open(f"{BCIPY_ROOT}/parameters/symbol_positions.json", 'r') as params_file:
                symbol_positions = json.load(params_file)
            # Subtract the symbol positions from the data:
            centralized_data_train.append(model.reshaper.centralize_all_data(train_dict[sym], symbol_positions[sym]))

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
            likelihoods, predictions = model.predict(
                test_dict[sym],
                np.squeeze(np.array(means_all)),
                np.squeeze(np.array(covs_all)))
            acc_all_symbols[sym] = calculate_acc(predictions, counter)
            accuracy += acc_all_symbols[sym]
            counter += 1
        accuracy /= counter

        df = pd.DataFrame(acc_all_symbols, index=[0])
        df.to_csv('my_data.csv', index=False)

        # Plot all accuracies as bar plot:
        figure_handles = visualize_gaze_accuracies(acc_all_symbols, accuracy, save_path=data_folder, show=show_figures)
        figures.append(figure_handles)

    if model_type == "Centralized":
        # Model 2: Fit Gaussian mixture on a centralized data
        all_data = np.concatenate(centralized_data_train, axis=0)

        # Visualize the results:
        figure_handles = visualize_centralized_data(
            all_data,
            save_path=data_folder if save_figures else None,
            show=show_figures,
            img_path=f"{data_folder}/matrix.png",
            raw_plot=True,
        )
        figures.append(figure_handles)

        model.fit(all_data)

        # Calculate means, covariances for each symbol.
        for sym in symbol_set:
            if len(test_dict[sym]) == 0:
                continue
            means, covs = model.evaluate(symbol_positions[sym])

            # Visualize the results:
            le = preprocessed_data[sym][0]
            re = preprocessed_data[sym][1]
            figure_handles = visualize_gaze_inquiries(
                le, re,
                means, covs,
                save_path=None,
                show=False,
                img_path=f"{data_folder}/matrix.png",
                raw_plot=True,
            )
            figures.append(figure_handles)
            left_eye_all.append(le)
            right_eye_all.append(re)
            means_all.append(means)
            covs_all.append(covs)

        # Compute scores for the test set.
        accuracy = 0
        acc_all_symbols = {}
        counter = 0
        for sym in symbol_set:
            if len(test_dict[sym]) == 0:
                # Continue if there is no test data for this symbol:
                acc_all_symbols[sym] = 0
                continue
            likelihoods, predictions = model.predict(
                test_dict[sym],
                np.squeeze(np.array(means_all)),
                np.squeeze(np.array(covs_all)))
            acc_all_symbols[sym] = calculate_acc(predictions, counter)
            accuracy += acc_all_symbols[sym]
            counter += 1
        accuracy /= counter

        # Plot all accuracies as bar plot:
        figure_handles = visualize_gaze_accuracies(acc_all_symbols, accuracy, save_path=data_folder, show=show_figures)
        
        # Save accuracies for later use as a csv file:
        df = pd.DataFrame(acc_all_symbols, index=[0])
        df.to_csv('my_data.csv', index=False)
        # basak = pd.read_csv('my_data.csv')
        
        figures.append(figure_handles)

    fig_handles = visualize_results_all_symbols(
        left_eye_all, right_eye_all,
        means_all, covs_all,
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
    data_file_paths = [
        path for path in (Path(data_folder, raw_data_filename(device_spec))
                          for device_spec in devices_by_name.values())
        if path.exists()
    ]

    models = []
    figure_handles = []
    for raw_data_path in data_file_paths:
        raw_data = load_raw_data(raw_data_path)
        device_spec = devices_by_name.get(raw_data.daq_type)
        # extract relevant information from raw data object eeg
        # if device_spec.content_type == "EEG":
        #     erp_model, erp_figure_handles = analyze_erp(
        #         raw_data, parameters, device_spec, data_folder, estimate_balanced_acc, save_figures, show_figures)
        #     models.append(erp_model)
        #     figure_handles.append(erp_figure_handles)

        if device_spec.content_type == "Eyetracker":
            et_model, et_figure_handles = analyze_gaze(
                raw_data, parameters, device_spec, data_folder, save_figures, show_figures, model_type="Individual")
            models.append(et_model)
            figure_handles.append(et_figure_handles)

    if alert_finished:
        play_sound(f"{STATIC_AUDIO_PATH}/{parameters['alert_sound_file']}")
    return models, figure_handles


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
    parser.set_defaults(balanced=False)
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
