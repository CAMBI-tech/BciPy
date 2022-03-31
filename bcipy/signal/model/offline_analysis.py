import logging
from pathlib import Path
from typing import Tuple
from bcipy.helpers.acquisition import analysis_channel_names_by_pos, analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from matplotlib.figure import Figure
from bcipy.helpers.stimuli import play_sound
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.triggers import trigger_decoder, TriggerType
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.process import get_default_transform, filter_inquiries
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s")


def subset_data(data, labels, test_size=0.2, random_state=0):
    data = data.swapaxes(0, 1)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    train_data = train_data.swapaxes(0, 1)
    test_data = test_data.swapaxes(0, 1)
    return train_data, test_data, train_labels, test_labels


@report_execution_time
def offline_analysis(
    data_folder: str = None, parameters: dict = {}, alert_finished: bool = True
) -> Tuple[SignalModel, Figure]:
    """Gets calibration data and trains the model in an offline fashion.
    pickle dumps the model into a .pkl folder
    Args:
        data_folder(str): folder of the data
            save all information and load all from this folder
        parameter(dict): parameters for running offline analysis
        alert_finished(bool): whether or not to alert the user offline analysis complete

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
    triggers_file = parameters.get("trigger_file_name", "triggers")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

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
    log.info(f"Device type: {type_amp}")
    log.info(
        f"Data processing settings: [Filter=[{filter_low}-{filter_high}]; order=[{filter_order}], "
        f"Notch=[{notch_filter}]], Downsample=[{downsample_rate}]]"
    )

    k_folds = parameters.get("k_folds")
    model = PcaRdaKdeModel(k_folds=k_folds)

    # Process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{triggers_file}.txt",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    inquiries, inquiry_labels, inquiry_timing = model.reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=raw_data.by_channel(),
        fs=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=poststim_length,
        prestimulus_length=prestim_length,
        transformation_buffer=buffer,
    )

    inquiries, fs = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(poststim_length * fs)
    data = model.reshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing, downsample_rate)

    # define the training classes using integers, where 0=nontargets/1=targets
    labels = [1 if label == "target" else 0 for label in trigger_targetness]

    # train and save the model as a pkl file
    log.info("Training model. This will take some time...")
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.fit(data, labels)
    log.info(f"Training complete [AUC={model.auc:0.4f}]. Saving data...")

    # # Experimenting with an 80/20 split and checking balanced accuracy
    # train_data, test_data, train_labels, test_labels = subset_data(data, labels)
    # model.fit(train_data, train_labels)
    # probs = model.predict_proba(test_data)
    # preds = probs.argmax(-1)
    # log.info(f"Balanced acc: {balanced_accuracy_score(test_labels, preds)}")

    model.save(data_folder + f"/model_{model.auc:0.4f}.pkl")

    figure_handles = visualize_erp(
        data,
        labels,
        fs,
        plot_average=False,  # set to True to see all channels target/nontarget averages
        save_path=data_folder,
        channel_names=analysis_channel_names_by_pos(channels, channel_map),
        show_figure=False,
        figure_name="average_erp.pdf",
    )
    if alert_finished:
        offline_analysis_tone = parameters.get("offline_analysis_tone")
        play_sound(offline_analysis_tone)

    return model, figure_handles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default=None)
    parser.add_argument("-p", "--parameters_file", default="bcipy/parameters/parameters.json")
    args = parser.parse_args()

    log.info(f"Loading params from {args.parameters_file}")
    parameters = load_json_parameters(args.parameters_file, value_cast=True)
    offline_analysis(args.data_folder, parameters, alert_finished=False)
    log.info("Offline Analysis complete.")
