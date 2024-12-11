# mypy: disable-error-code="attr-defined"
import json
import logging
import subprocess
from pathlib import Path
from typing import List

import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

import bcipy.acquisition.devices as devices
from bcipy.acquisition.devices import DeviceSpec
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME,
                          DEFAULT_PARAMETERS_PATH, DEFAULT_DEVICES_PATH,
                          TRIGGER_FILENAME, SESSION_LOG_FILENAME,
                          STIMULI_POSITIONS_FILENAME)
from bcipy.helpers.acquisition import analysis_channels, raw_data_filename
from bcipy.helpers.load import (load_experimental_data, load_json_parameters,
                                load_raw_data)
from bcipy.gui.alert import confirm
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import save_model
from bcipy.helpers.stimuli import update_inquiry_timing
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.raw_data import RawData
from bcipy.preferences import preferences
from bcipy.signal.model.base_model import SignalModel, SignalModelMetadata
from bcipy.signal.model.gaussian_mixture import (GazeModelResolver)
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import (ERPTransformParams, extract_eye_info,
                                  filter_inquiries, get_default_transform)
from bcipy.signal.model.evaluate.fusion import calculate_eeg_gaze_fusion_acc

log = logging.getLogger(SESSION_LOG_FILENAME)
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


def analyze_erp(
        erp_data: RawData,
        parameters: Parameters,
        device_spec: DeviceSpec,
        data_folder: str,
        estimate_balanced_acc: bool,
        save_figures: bool=False,
        show_figures: bool=False) -> SignalModel:
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
    trial_window = parameters.get("trial_window", (0.0, 0.5))
    window_length = trial_window[1] - trial_window[0]

    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)

    # Get signal filtering information
    transform_params = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate
    static_offset = device_spec.static_offset

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
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
        offset=static_offset,
        remove_pre_fixation=True,
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
    try:
        model.fit(data, labels)
        model.metadata = SignalModelMetadata(device_spec=device_spec,
                                             transform=default_transform,
                                             evidence_type="ERP",
                                             auc=model.auc)
        log.info(f"Training complete [AUC={model.auc:0.4f}]. Saving data...")
    except Exception as e:
        log.error(f"Error training model: {e}")

    try:
        # Using an 80/20 split, report on balanced accuracy
        if estimate_balanced_acc:
            train_data, test_data, train_labels, test_labels = subset_data(data, labels, test_size=0.2)
            dummy_model = PcaRdaKdeModel(k_folds=k_folds)
            dummy_model.fit(train_data, train_labels)
            probs = dummy_model.predict_proba(test_data)
            preds = probs.argmax(-1)
            score = balanced_accuracy_score(test_labels, preds)
            log.info(f"Balanced acc with 80/20 split: {score}")
            model.metadata.balanced_accuracy = score
            del dummy_model, train_data, test_data, train_labels, test_labels, probs, preds

    except Exception as e:
        log.error(f"Error calculating balanced accuracy: {e}")

    save_model(model, Path(data_folder, f"model_{device_spec.content_type.lower()}_{model.auc:0.4f}.pkl"))
    preferences.signal_model_directory = data_folder

    if save_figures or show_figures:
        cmd = f'bcipy-erp-viz --session_path "{data_folder}" --parameters "{parameters["parameter_location"]}"'
        if save_figures:
            cmd += ' --save'
        if show_figures:
            cmd += ' --show'
        subprocess.run(
            cmd,
            shell=True
        )
    return model


def analyze_gaze(
        gaze_data: RawData,
        parameters: Parameters,
        device_spec: DeviceSpec,
        data_folder: str,
        model_type: str="GaussianProcess",
        symbol_set: List[str] = alphabet()) -> SignalModel:
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
        model_type (str): Type of gaze model to be used. Options are: "GMIndividual", "GMCentralized",
        or "GaussianProcess".
    """
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

    model = GazeModelResolver.resolve(model_type)

    # Extract all Triggers info
    _trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
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

    target_symbols = trigger_symbols[0::stim_size + 1]  # target symbols are the PROMPT triggers
    # Use trigger_timing to generate time windows for each letter flashing
    # Take every 10th trigger as the start point of timing.
    inq_start = trigger_timing[1::stim_size + 1]  # start of each inquiry (here we jump over prompts)

    # Extract the inquiries dictionary with keys as target symbols and values as inquiry windows:
    inquiries_dict, inquiries_list, _ = model.reshaper(
        inq_start_times=inq_start,
        target_symbols=target_symbols,
        gaze_data=data,
        sample_rate=sample_rate,
        stimulus_duration=flash_time,
        num_stimuli_per_inquiry=stim_size,
        symbol_set=symbol_set,
    )

    # Apply preprocessing:
    inquiry_length = inquiries_list[0].shape[1]  # number of time samples in each inquiry
    predefined_dimensions = 4  # left_x, left_y, right_x, right_y
    preprocessed_array = np.zeros((len(inquiries_list), predefined_dimensions, inquiry_length))
    # Extract left_x, left_y, right_x, right_y for each inquiry
    for j in range(len(inquiries_list)):
        left_eye, right_eye, _, _, _, _ = extract_eye_info(inquiries_list[j])
        preprocessed_array[j] = np.concatenate((left_eye.T, right_eye.T,), axis=0)

    preprocessed_data = {i: [] for i in symbol_set}
    for i in symbol_set:
        # Skip if there's no evidence for this symbol:
        if len(inquiries_dict[i]) == 0:
            continue

        for j in range(len(inquiries_dict[i])):
            left_eye, right_eye, _, _, _, _ = extract_eye_info(inquiries_dict[i][j])
            preprocessed_data[i].append((np.concatenate((left_eye.T, right_eye.T), axis=0)))
            # Inquiries x All Dimensions (left_x, left_y, right_x, right_y) x Time
        preprocessed_data[i] = np.array(preprocessed_data[i])

    centralized_data = {i: [] for i in symbol_set}
    time_average = {i: [] for i in symbol_set}

    for sym in symbol_set:
        # Skip if there's no evidence for this symbol:
        try:
            if len(inquiries_dict[sym]) == 0:
                continue

        except BaseException:
            log.info(f"No evidence from symbol {sym}")
            continue

        # Fit the model based on model type.
        if model_type == "GMIndividual":
            # Model 1: Fit Gaussian mixture on each symbol separately
            reshaped_data = preprocessed_data[sym].reshape(
                (preprocessed_data[sym].shape[0] *
                 preprocessed_data[sym].shape[2],
                 preprocessed_data[sym].shape[1]))
            model.fit(reshaped_data)

        if model_type == "GMCentralized":
            # Centralize the data using symbol positions and fit a single Gaussian.
            # Load json file.
            with open(f"{data_folder}/{STIMULI_POSITIONS_FILENAME}", 'r') as params_file:
                symbol_positions = json.load(params_file)

            # Subtract the symbol positions from the data:
            for j in range(len(preprocessed_data[sym])):
                centralized_data[sym].append(model.centralize(preprocessed_data[sym][j], symbol_positions[sym]))

        if model_type == "GaussianProcess":
            # Instead of centralizing, take the time average:
            for j in range(len(preprocessed_data[sym])):
                temp = np.mean(preprocessed_data[sym][j], axis=1)
                time_average[sym].append(temp)
                centralized_data[sym].append(
                    model.subtract_mean(
                        preprocessed_data[sym][j],
                        temp))  # Delta_t = X_t - mu
            centralized_data[sym] = np.array(centralized_data[sym])
            time_average[sym] = np.mean(np.array(time_average[sym]), axis=0)

    if model_type == "GaussianProcess":
        # Split the data into train and test sets & fit the model:
        centralized_gaze_data = np.zeros_like(preprocessed_array)
        for i, (_, sym) in enumerate(zip(preprocessed_array, target_symbols)):
            centralized_gaze_data[i] = model.subtract_mean(preprocessed_array[i], time_average[sym])
        reshaped_data = centralized_gaze_data.reshape(
            (len(centralized_gaze_data), inquiry_length * predefined_dimensions))

        cov_matrix = np.cov(reshaped_data, rowvar=False)
        time_horizon = 9

        for eye_coord_0 in range(predefined_dimensions):
            for eye_coord_1 in range(predefined_dimensions):
                for time_0 in range(inquiry_length):
                    for time_1 in range(inquiry_length):
                        l_ind = eye_coord_0 * inquiry_length + time_0
                        m_ind = eye_coord_1 * inquiry_length + time_1
                        if np.abs(time_1 - time_0) > time_horizon:
                            cov_matrix[l_ind, m_ind] = 0
        reshaped_mean = np.mean(reshaped_data, axis=0)

        # Save model parameters which are mean and covariance matrix
        model.fit(reshaped_mean)

    if model_type == "GMCentralized":
        # Fit the model parameters using the centralized data:
        # flatten the dict to a np array:
        cent_data = np.concatenate([centralized_data[sym] for sym in symbol_set], axis=0)
        # Merge the first and third dimensions:
        cent_data = cent_data.reshape((cent_data.shape[0] * cent_data.shape[2], cent_data.shape[1]))

        # cent_data = np.concatenate(centralized_data, axis=0)
        model.fit(cent_data)

    model.metadata = SignalModelMetadata(device_spec=device_spec,
                                         transform=None,
                                         acc=model.acc)
    log.info("Training complete for Eyetracker model. Saving data...")
    save_model(
        model,
        Path(data_folder, f"model_{device_spec.content_type.lower()}_{model.acc}.pkl"))
    return model


@report_execution_time
def offline_analysis(
    data_folder: str = None,
    parameters: Parameters = None,
    alert: bool = True,
    estimate_balanced_acc: bool = False,
    show_figures: bool = False,
    save_figures: bool = False,
    n_iterations: int = 10
) -> List[SignalModel]:
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

    Parameters:
    ----------
        data_folder(str): folder of the data
            save all information and load all from this folder
        parameter(dict): parameters for running offline analysis
        alert(bool): whether or not to alert the user offline analysis steps and completion
        estimate_balanced_acc(bool): if true, uses another model copy on an 80/20 split to
            estimate balanced accuracy
        show_figures(bool): if true, shows ERP figures after training
        save_figures(bool): if true, saves ERP figures after training to the data folder

    Returns:
    --------
        model (SignalModel): trained model
    """
    assert parameters, "Parameters are required for offline analysis."
    if not data_folder:
        data_folder = load_experimental_data()

    # Load default devices which are used for training the model with different channels, etc.
    devices_by_name = devices.load(
        Path(DEFAULT_DEVICES_PATH, DEFAULT_DEVICE_SPEC_FILENAME), replace=True)

    # Load the active devices used during a session; this will be used to exclude inactive devices
    active_devices_by_name = devices.load(
        Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME), replace=True)
    active_devices = (spec for spec in active_devices_by_name.values()
                      if spec.is_active)
    active_raw_data_paths = (Path(data_folder, raw_data_filename(device_spec))
                             for device_spec in active_devices)
    data_file_paths = [str(path) for path in active_raw_data_paths if path.exists()]

    num_devices = len(data_file_paths)
    assert num_devices >= 1 and num_devices < 3, (
        f"Offline analysis requires at least one data file and at most two data files. Found: {num_devices}"
    )

    symbol_set = alphabet()
    fusion = False
    if num_devices == 2:
        # Ensure there is an EEG and Eyetracker device
        fusion = True
        log.info("Fusion analysis enabled.")

        if alert:
            if not confirm("Starting fusion analysis... Hit cancel to train models individually."):
                fusion = False

        if fusion:
            eeg_data = load_raw_data(data_file_paths[0])
            device_spec_eeg = devices_by_name.get(eeg_data.daq_type)
            assert device_spec_eeg.content_type == "EEG", "First device must be EEG"
            gaze_data = load_raw_data(data_file_paths[1])
            device_spec_gaze = devices_by_name.get(gaze_data.daq_type)
            assert device_spec_gaze.content_type == "Eyetracker", "Second device must be Eyetracker"
            eeg_acc, gaze_acc, fusion_acc = calculate_eeg_gaze_fusion_acc(
                eeg_data,
                gaze_data,
                device_spec_eeg,
                device_spec_gaze,
                symbol_set,
                parameters,
                data_folder,
                n_iterations=n_iterations,
            )

            log.info(f"EEG Accuracy: {eeg_acc}, Gaze Accuracy: {gaze_acc}, Fusion Accuracy: {fusion_acc}")

    # Ask the user if they want to proceed with full dataset model training
    models = []
    log.info(f"Starting offline analysis for {data_file_paths}")
    for raw_data_path in data_file_paths:
        raw_data = load_raw_data(raw_data_path)
        device_spec = devices_by_name.get(raw_data.daq_type)
        # extract relevant information from raw data object eeg

        if device_spec.content_type == "EEG" and device_spec.is_active:
            erp_model = analyze_erp(
                raw_data,
                parameters,
                device_spec,
                data_folder,
                estimate_balanced_acc,
                save_figures,
                show_figures)
            models.append(erp_model)

        if device_spec.content_type == "Eyetracker" and device_spec.is_active:
            et_model = analyze_gaze(
                raw_data,
                parameters,
                device_spec,
                data_folder,
                symbol_set=symbol_set)
            models.append(et_model)

    if alert:
        log.info("Alerting Offline Analysis Complete")
        results = [f"{model.name}: {model.auc}" for model in models]
        confirm(f"Offline analysis complete! \n Results={results}")
    log.info("Offline analysis complete")
    return models


def main():
    """Main function for offline analysis client.

    Parses command line arguments and runs offline analysis.

    Command Line Arguments:

    -d, --data_folder: Path to the folder containing the data to be analyzed.
    -p, --parameters_file: Path to the parameters file. Default is DEFAULT_PARAMETERS_PATH.
    -s, --save_figures: If true, saves data figures after training to the data folder.
    -v, --show_figures: If true, shows data figures after training.
    --alert: If true, alerts the user when offline analysis is complete.
    --balanced-acc: If true, uses another model copy on an 80/20 split to estimate balanced accuracy.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folder",
        default=None,
        help="Path to the folder containing the BciPy data to be analyzed.")
    parser.add_argument(
        "-p",
        "--parameters_file",
        default=DEFAULT_PARAMETERS_PATH,
        help="Path to the BciPy parameters file.")
    parser.add_argument("-s", "--save_figures", action="store_true", help="Save figures after training.")
    parser.add_argument("-v", "--show_figures", action="store_true", help="Show figures after training.")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations for fusion analysis.")
    parser.add_argument(
        "--alert",
        dest="alert",
        action="store_true",
        help="Alert the user when offline analysis is complete.")
    parser.add_argument("--balanced-acc", dest="balanced", action="store_true")
    parser.set_defaults(alert=False)
    parser.set_defaults(balanced=False)
    parser.set_defaults(save_figures=False)
    parser.set_defaults(show_figures=False)
    args = parser.parse_args()

    log.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    log.info(
        f"Starting offline analysis client with the following: Data={args.data_folder} || "
        f"Save Figures={args.save_figures} || Show Figures={args.show_figures} || "
        f"Alert={args.alert} || Calculate Balanced Accuracy={args.balanced}")

    offline_analysis(
        args.data_folder,
        parameters,
        alert=args.alert,
        estimate_balanced_acc=args.balanced,
        save_figures=args.save_figures,
        show_figures=args.show_figures,
        n_iterations=args.iterations)


if __name__ == "__main__":
    main()
