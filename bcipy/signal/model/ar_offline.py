import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from bcipy.config import DEFAULT_PARAMETERS_PATH, TRIGGER_FILENAME, RAW_DATA_FILENAME, DEFAULT_DEVICE_SPEC_FILENAME
# from bcipy.preferences import preferences
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.convert import convert_to_mne
from bcipy.helpers.stimuli import play_sound, mne_epochs
from bcipy.helpers.system_utils import report_execution_time
import bcipy.acquisition.devices as devices
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import filter_inquiries, get_default_transform
from matplotlib.figure import Figure
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import mne

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")

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
        data, labels, test_size=test_size, random_state=random_state, stratify=labels,
    )
    train_data = train_data.swapaxes(0, 1)
    test_data = test_data.swapaxes(0, 1)
    return train_data, test_data, train_labels, test_labels

@report_execution_time
def offline_analysis(
    data_folder: str = None,
    parameters: dict = {},
    mne_data: mne.io.Raw = None,
    baseline: Tuple[float, float] = (0, 0),
    drop_bad_epochs: bool = False,
    estimate_balanced_acc: bool = False,
) -> tuple:
    """Gets calibration data and trains the model in an offline fashion.
    pickle dumps the model into a .pkl folder
    Args:
        data_folder(str): folder of the data
            save all information and load all from this folder
        parameter(dict): parameters for running offline analysis
        mne_data(mne.io.Raw): mne data object
        baseline(tuple): baseline for the ERP
        drop_bad_epochs(bool): whether or not to drop bad epochs

    How it Works:
    - reads data and information from a .csv calibration file
    - reads trigger information from a .txt trigger file
    - filters data
    - reshapes and labels the data for the training procedure
    - fits the model to the data
        - uses cross validation to select parameters
        - based on the parameters, trains system using all the data
    - pickle dumps model into .pkl file
    - generates and saves ERP figure
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
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    log.info(
        f"\nData processing settings for [{data_folder}]: \n"
        f"Filter: [{filter_low}-{filter_high}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {prestim_length}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    devices.load(Path(data_folder, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(raw_data.daq_type)

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

    k_folds = parameters.get("k_folds") # by default this is 10
    model = PcaRdaKdeModel(k_folds=k_folds)

    # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, device_spec)
    data, _ = raw_data.by_channel()

    if not mne_data:
        # mne data
        mne_data, _ = convert_to_mne(raw_data, channel_map, transform=default_transform)

    # print(poststim_length)
    epochs = mne_epochs(
        mne_data,
        trigger_timing,
        trigger_targetness,
        interval=[0, poststim_length],
        baseline=baseline,
        reject_by_annotation=drop_bad_epochs)

    if baseline[0] > 0:
        # apply baseline correction
        epochs.apply_baseline(baseline=baseline)
    
    data = epochs.get_data() # (1100, 19, 76) epochs, channels, samples
    # map to channels, epochs, samples
    # data = np.transpose(data, (1, 0, 2)) # (19, 1100, 76) channels, epochs, samples
    # data = np.transpose(data, (1, 0, 2)) # (1100, 19, 76) epochs, channels, samples
    # breakpoint()
    # define the training classes using integers, where 0=nontargets/1=targets
    labels = []
    for i in range(len(epochs)):
        try:
            epochs[i].event_id['1']
            labels.append(0)
        except:
            labels.append(1)

    # breakpoint() for getting counts of labels after artifact rejection
    nontarget = labels.count(0)
    target = labels.count(1)

    # get the count of the original labels
    nontarget_orig = trigger_targetness.count('nontarget')
    target_orig = trigger_targetness.count('target')
    print(f"nontarget: {nontarget}, target: {target}")

    if nontarget < 500 or target < 50:
        log.info("Not enough data to train model")
        breakpoint()
        raise Exception("Not enough data to train model")

    return data, labels, {'nontarget': nontarget, 'target': target, 'nontarget_orig': nontarget_orig, 'target_orig': target_orig}

    # train and save the model as a pkl file
    log.info("Training model. This will take some time...")
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.fit(data, labels)
    log.info(f"Training complete [AUC={model.auc:0.4f}]. Saving data...")

    model.save(f"{data_folder}/re_model_fulldatasetfilter_{model.auc:0.4f}.pkl")

    if estimate_balanced_acc:
        train_data, test_data, train_labels, test_labels = subset_data(data, labels, test_size=0.2)
        dummy_model = PcaRdaKdeModel(k_folds=k_folds)
        dummy_model.fit(train_data, train_labels)
        probs = dummy_model.predict_proba(test_data)
        preds = probs.argmax(-1)
        ba_score = balanced_accuracy_score(test_labels, preds)
        log.info(f"Balanced acc with 80/20 split: {ba_score:0.4f}")
        del dummy_model, train_data, test_data, train_labels, test_labels, probs, preds

        return model, ba_score
    
    return target, nontarget


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    parser.add_argument("-p", "--parameters_file", default=DEFAULT_PARAMETERS_PATH)
    parser.add_argument("--alert", dest="alert", action="store_true")
    parser.set_defaults(alert=False)
    args = parser.parse_args()

    log.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)

    offline_analysis(data_folder=args.data_folder, parameters=parameters, alert_finished=args.alert)
    log.info("AR Offline Analysis complete.")
