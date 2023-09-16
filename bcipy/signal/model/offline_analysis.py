import logging
from pathlib import Path
from typing import Tuple

from bcipy.config import (DEFAULT_PARAMETERS_PATH, TRIGGER_FILENAME,
                          RAW_DATA_FILENAME, STATIC_AUDIO_PATH,
                          DEFAULT_DEVICE_SPEC_FILENAME,
                          EYE_TRACKER_FILENAME_PREFIX, STATIC_IMAGES_PATH)
from bcipy.preferences import preferences
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.stimuli import play_sound, update_inquiry_timing
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.triggers import TriggerType, trigger_decoder, TriggerHandler
from bcipy.helpers.visualization import visualize_erp, visualize_gaze, visualize_gaze_inquiries
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.model.fusion_model import GazeModel
from bcipy.signal.process import filter_inquiries, get_default_transform
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import bcipy.acquisition.devices as devices

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

DIPSIZE = (1707, 1067)
IMG_PATH = f'{STATIC_IMAGES_PATH}/main/matrix_grid.png'
# IMG_PATH = f'{STATIC_IMAGES_PATH}/main/rsvp.png'

def subset_data(data: np.ndarray, labels: np.ndarray, test_size: float, random_state=0):
    """Performs a train/test split on the provided data and labels, accounting for
    the current shape convention (channel dimension in front, instead of batch dimension in front).

    Args:
        data (np.ndarray): Shape (channels, items, time)
        labels (np.ndarray): Shape (items,)
        test_size (float): fraction of data to be used for testing
        random_state (int, optional): fixed random seed

    Returns:
        train_data (np.ndarray): Shape (channels, train_items, time)
        test_data (np.ndarray): Shape (channels, test_items, time)
        train_labels (np.ndarray): Shape (train_items,)
        test_labels (np.ndarray): Shape (test_items,)
    """
    data = data.swapaxes(0, 1)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    train_data = train_data.swapaxes(0, 1)
    test_data = test_data.swapaxes(0, 1)
    return train_data, test_data, train_labels, test_labels


@report_execution_time
def offline_analysis(
    data_folder: str = None,
    parameters: dict = {},
    alert_finished: bool = True,
    estimate_balanced_acc: bool = False,
    show_figures: bool = False,
    save_figures: bool = False,
) -> Tuple[SignalModel, Figure]:
    """Gets calibration data and trains the model in an offline fashion.
    pickle dumps the model into a .pkl folder
    Args:
        data_folder(str): folder of the data
            save all information and load all from this folder
        parameter(dict): parameters for running offline analysis
        alert_finished(bool): whether or not to alert the user offline analysis complete
        estimate_balanced_acc(bool): if true, uses another model copy on an 80/20 split to
            estimate balanced accuracy
        show_figures(bool): if true, shows ERP figures after training
        save_figures(bool): if true, saves ERP figures after training to the data folder

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
    """

    if not data_folder:
        data_folder = load_experimental_data()

    # extract relevant session information from parameters file
    poststim_length = parameters.get("trial_length")
    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)

    acq_mode = parameters.get("acq_mode")
    # handle multiple acquisition. Note: we only currently support Eyetracker + EEG
    data_file_paths = []
    if '+' in acq_mode:
        if 'eyetracker' in acq_mode.lower():
            # find the eyetracker data file with the prefix of EYE_TRACKER_FILENAME_PREFIX
            eye_tracker_file = [ f.name for f in Path(data_folder).iterdir() if f.name.startswith(EYE_TRACKER_FILENAME_PREFIX)]
            assert len(eye_tracker_file) == 1, f"Found {len(eye_tracker_file)} eyetracker files in {data_folder}. Expected 1."
            data_file_paths.extend(eye_tracker_file)
        else:
            raise ValueError(f"Unsupported acquisition mode: {acq_mode}. Eyetracker must be included.")
        if 'eeg' in acq_mode.lower():
            # find the eeg data file with the prefix of EEG_FILENAME_PREFIX
            eeg_file = [ f.name for f in Path(data_folder).iterdir() if f.name.startswith(RAW_DATA_FILENAME)]
            assert len(eeg_file) == 1, f"Found {len(eeg_file)} EEG files in {data_folder}. Expected 1."
            data_file_paths.extend(eeg_file)
        else:
            raise ValueError(f"Unsupported acquisition mode: {acq_mode}. EEG must be included.")
    else:
        data_file_paths = [f"{RAW_DATA_FILENAME}.csv"]

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    log.info(
        f"\nData processing settings: \n"
        f"Filter: [{filter_low}-{filter_high}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {prestim_length}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data: [RawData(Tobii-P0), RawData(DSI-24)]
    raw_data = load_raw_data(data_folder, data_file_paths)

    for mode_data in raw_data:
        devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
        device_spec = devices.preconfigured_device(mode_data.daq_type)
        # extract relevant information from raw data object eeg
        if device_spec.content_type == "EEG":
            channels = mode_data.channels
            type_amp = mode_data.daq_type
            sample_rate = mode_data.sample_rate

            # setup filtering
            default_transform = get_default_transform(
                sample_rate_hz=sample_rate,
                notch_freq_hz=notch_filter,
                bandpass_low=filter_low,
                bandpass_high=filter_high,
                bandpass_order=filter_order,
                downsample_factor=downsample_rate,
            )

            log.info(f"Channels read from csv: {channels}")
            log.info(f"Device type: {type_amp}, fs={sample_rate}")

            k_folds = parameters.get("k_folds")
            model = PcaRdaKdeModel(k_folds=k_folds)

            # Process triggers.txt files
            trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
                offset=static_offset,
                trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
                exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
            )
            # Channel map can be checked from data.csv file or the devices.json located in the acquisition module
            # The timestamp column [0] is already excluded.
            channel_map = analysis_channels(channels, device_spec)
            channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
            log.info(f'Channels used in analysis: {channels_used}')

            data, fs = mode_data.by_channel()

            inquiries, inquiry_labels, inquiry_timing = model.reshaper(
                trial_targetness_label=trigger_targetness,
                timing_info=trigger_timing,
                eeg_data=data,
                sample_rate=sample_rate,
                trials_per_inquiry=trials_per_inquiry,
                channel_map=channel_map,
                poststimulus_length=poststim_length,
                prestimulus_length=prestim_length,
                transformation_buffer=buffer,
            )

            inquiries, fs = filter_inquiries(inquiries, default_transform, sample_rate)
            inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
            trial_duration_samples = int(poststim_length * fs)
            trial_data = model.reshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

            # define the training classes using integers, where 0=nontargets/1=targets
            labels = inquiry_labels.flatten()

            # train and save the model as a pkl file
            log.info("Training EEG model. This will take some time...")
            model = PcaRdaKdeModel(k_folds=k_folds)
            model.fit(trial_data, labels)
            log.info(f"Training EEG complete [AUC={model.auc:0.4f}]. Saving data...")

            model.save(data_folder + f"/model_{model.auc:0.4f}.pkl")
            preferences.signal_model_directory = data_folder

            # Using an 80/20 split, report on balanced accuracy
            if estimate_balanced_acc:
                train_data, test_data, train_labels, test_labels = subset_data(trial_data, labels, test_size=0.2)
                dummy_model = PcaRdaKdeModel(k_folds=k_folds)
                dummy_model.fit(train_data, train_labels)
                probs = dummy_model.predict_proba(test_data)
                preds = probs.argmax(-1)
                score = balanced_accuracy_score(test_labels, preds)
                log.info(f"Balanced acc with 80/20 split: {score}")
                del dummy_model, train_data, test_data, train_labels, test_labels, probs, preds

            figure_handles = visualize_erp(
                mode_data,
                channel_map,
                trigger_timing,
                labels,
                poststim_length,
                transform=default_transform,
                plot_average=True,
                plot_topomaps=True,
                save_path=data_folder if save_figures else None,
                show=show_figures
            )
        if device_spec.content_type == "Eyetracker":
            print(device_spec)
            figure_handles = visualize_gaze(
                mode_data,
                save_path=data_folder if save_figures else None,
                show=True,
                raw_plot=True,
            )
           
            channels = mode_data.channels
            type_amp = mode_data.daq_type
            sample_rate = mode_data.sample_rate

            log.info(f"Channels read from csv: {channels}")
            log.info(f"Device type: {type_amp}, fs={sample_rate}")       
            channel_map = analysis_channels(channels, device_spec)

            channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
            log.info(f'Channels used in analysis: {channels_used}')

            data, fs = mode_data.by_channel()

            model = GazeModel()

            # Process triggers.txt files (again!)
            trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
                remove_pre_fixation = False,
                apply_system_offset=False,
                trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
                exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION, TriggerType.SYSTEM, TriggerType.OFFSET]
            )
            ''' Trigger_timing includes PROMPT and excludes FIXATION '''
           
            # Use trigger_timing to generate time windows for each letter flashing
            # Take every 10th trigger as the start point of timing.
            # trigger_targetness keeps the PROMPT info, use it to find the target symbol.
            target_symbols = trigger_symbols[0::11]  # target symbols
            inq_start = trigger_timing[1::11]  # start of each inquiry (here we jump over prompts)

            # Find the inquiries starting by the inq_start_times, (9, 100, 180):           
            inquiries = model.reshaper(
            trial_targetness_label=trigger_targetness,
            inq_start_times=inq_start,
            target_symbols = target_symbols,
            gaze_data=data,
            sample_rate=sample_rate,
            trials_per_inquiry=trials_per_inquiry,
            poststimulus_length=poststim_length,
            prestimulus_length=prestim_length,
            transformation_buffer=buffer
            )

            symbol_set = alphabet()

            # Extract the data for each target label and each eye separately. 
            # Apply preprocessing:
            preprocessed_data = {i: [] for i in symbol_set}
            for i in symbol_set:
                # Skip if there's no evidence for this symbol:
                if len(inquiries[i]) == 0:
                    continue
                
                left_eye, right_eye = model.reshaper.extract_eye_info(inquiries[i])
                preprocessed_data[i] = np.array([left_eye, right_eye])    # Channels x Sample Size x Dimensions(x,y)

                # Train test split:
                test_size = int(len(right_eye) * 0.2)
                train_size = len(right_eye) - test_size
                train_right_eye = right_eye[:train_size]
                test_right_eye = right_eye[train_size:]

                train_left_eye = left_eye[:train_size]
                test_left_eye = left_eye[train_size:]

                # Fit the model:
                model.fit(train_right_eye)

                scores, means, covs = model.get_scores(test_right_eye)
                # print(scores)

                # Visualize the results:
                figure_handles = visualize_gaze_inquiries(
                    left_eye, right_eye,
                    means, covs,
                    save_path=None,
                    show=True,
                    raw_plot=True,
                )                     
            

    if alert_finished:
        play_sound(f"{STATIC_AUDIO_PATH}/{parameters['alert_sound_file']}")
    return model, figure_handles


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
    parser.set_defaults(save_figures=False)
    parser.set_defaults(show_figures=False)
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
