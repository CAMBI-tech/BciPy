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
from bcipy.helpers.stimuli import play_sound, InquiryReshaper
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.triggers import trigger_decoder, TriggerType
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.process import get_default_transform


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s')

def load_model(model_path: Path, k_folds: int):
    """Load the PcaRdaKdeModel model at the given path"""
    model = PcaRdaKdeModel(k_folds=k_folds)
    model.load(model_path)
    return model


# @report_execution_time
# def offline_analysis(data_folder: str = None,
#                      parameters: dict = {}, alert_finished: bool = True) -> Tuple[SignalModel, Figure]:
#     """ Gets calibration data and trains the model in an offline fashion.
#         pickle dumps the model into a .pkl folder
#         Args:
#             data_folder(str): folder of the data
#                 save all information and load all from this folder
#             parameter(dict): parameters for running offline analysis
#             alert_finished(bool): whether or not to alert the user offline analysis complete

#         How it Works:
#         - reads data and information from a .csv calibration file
#         - reads trigger information from a .txt trigger file
#         - filters data
#         - reshapes and labels the data for the training procedure
#         - fits the model to the data
#             - uses cross validation to select parameters
#             - based on the parameters, trains system using all the data
#         - pickle dumps model into .pkl file
#         - generates and saves ERP figure
#         - [optional] alert the user finished processing
#     """

#     if not data_folder:
#         data_folder = load_experimental_data()

#     # extract relevant session information from parameters file
#     trial_length = parameters.get('trial_length')
#     trials_per_inquiry = parameters.get('stim_length')
#     triggers_file = parameters.get('trigger_file_name', 'triggers')
#     raw_data_file = parameters.get('raw_data_name', 'raw_data.csv')

#     # get signal filtering information
#     downsample_rate = parameters.get('down_sampling_rate')
#     notch_filter = parameters.get('notch_filter_frequency')
#     hp_filter = parameters.get('filter_high')
#     lp_filter = parameters.get('filter_low')
#     filter_order = parameters.get('filter_order')

#     # get offset and k folds
#     static_offset = parameters.get('static_trigger_offset', 0.0)
#     k_folds = parameters.get('k_folds')

#     # Load raw data
#     raw_data = load_raw_data(Path(data_folder, raw_data_file))
#     channels = raw_data.channels
#     type_amp = raw_data.daq_type
#     fs = raw_data.sample_rate

#     log.info(f'Channels read from csv: {channels}')
#     log.info(f'Device type: {type_amp}')

#     default_transform = get_default_transform(
#         sample_rate_hz=fs,
#         notch_freq_hz=notch_filter,
#         bandpass_low=lp_filter,
#         bandpass_high=hp_filter,
#         bandpass_order=filter_order,
#         downsample_factor=downsample_rate,
#     )
#     data, fs = default_transform(raw_data.by_channel(), fs)

#     # Process triggers.txt
#     trigger_values, trigger_timing, _ = trigger_decoder(
#         offset=static_offset,
#         trigger_path=f'{data_folder}/{triggers_file}.txt')

#     # Channel map can be checked from raw_data.csv file.
#     # The timestamp column is already excluded.
#     channel_map = analysis_channels(channels, type_amp)

#     model = PcaRdaKdeModel(k_folds=k_folds)
#     data, labels = model.reshaper(
#         trial_labels=trigger_values,
#         timing_info=trigger_timing,
#         eeg_data=data,
#         fs=fs,
#         trials_per_inquiry=trials_per_inquiry,
#         channel_map=channel_map,
#         trial_length=trial_length)

#     log.info('Training model. This will take some time...')
#     model.fit(data, labels)
#     # model_performance = model.evaluate(data, labels)

#     log.info(f'Training complete [AUC={model.auc:0.4f}]. Saving data...')

#     model.save(data_folder + f'/model_{model.auc:0.4f}.pkl')

#     figure_handles = visualize_erp(
#         data,
#         labels,
#         fs,
#         plot_average=False,  # set to True to see all channels target/nontarget
#         save_path=data_folder,
#         channel_names=analysis_channel_names_by_pos(channels, channel_map),
#         show_figure=False,
#         figure_name='average_erp.pdf'
#     )
#     if alert_finished:
#         offline_analysis_tone = parameters.get('offline_analysis_tone')
#         play_sound(offline_analysis_tone)

#     return model, figure_handles

def filter_inquiries(inquiries, transform, sample_rate):
    old_shape = inquiries.shape  # (C, I, 699)
    inq_flatten = inquiries.reshape(-1, old_shape[-1])  # (C*I, 699)
    inq_flatten_filtered, transformed_sample_rate = transform(inq_flatten, sample_rate)
    inquiries = inq_flatten_filtered.reshape(*old_shape[:2], inq_flatten_filtered.shape[-1])  # (C, I, ...)
    return inquiries, transformed_sample_rate


@report_execution_time
def filter_inquiry_and_get_trials(data_folder, parameters):
    """Try running a previous model as follows. Shouldn't get errors.

    # 2a (real, online model): raw -> inquiries -> filter -> trials -> model preds
    # 2b (trying to replay):  raw data -> InquiryReshaper -> filtered -> TrialReshaper

    raw data -> inquiries -> filter -> trials -> run model
    """

    # extract relevant session information from parameters file
    trial_length = parameters.get("trial_length")
    trials_per_inquiry = parameters.get("stim_length")
    # time_flash = parameters.get("time_flash")
    triggers_file = parameters.get("trigger_file_name", "triggers")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    hp_filter = parameters.get("filter_high")
    lp_filter = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")
    k_folds = parameters.get("k_folds")

    model = PcaRdaKdeModel(k_folds=k_folds)

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    log.info(f"Channels read from csv: {channels}")
    log.info(f"Device type: {type_amp}")

    # Process triggers.txt
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{triggers_file}.txt",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
    )
    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    # data , [0: 1, 0, 1], [0, ... , n]
    inquiries, inquiry_labels, inquiry_timing = InquiryReshaper()(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=raw_data.by_channel(),
        fs=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=trial_length,
        prestimulus_length=0.5,
        transformation_buffer=0.5,  # use this to add time to the end of the Inquiry for processing!
    )

    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=2,
        bandpass_high=45,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    inquiries, transformed_sample_rate = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(trial_length * transformed_sample_rate)
    trials = InquiryReshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing, downsample_rate)
    return model, trials, trigger_targetness, data_folder, transformed_sample_rate, channels, channel_map

@report_execution_time
def train(model, data, labels, folder, fs, channels, channel_map):

    labels = [0 if label == 'nontarget' else 1 for label in labels]
    model.fit(data, labels)
    log.info(f'Training complete [AUC={model.auc:0.4f}]. Saving data...')

    model.save(folder + f'/model_{model.auc:0.4f}.pkl')

    figure_handles = visualize_erp(
        data,
        labels,
        fs,
        plot_average=False,  # set to True to see all channels target/nontarget
        save_path=folder,
        channel_names=analysis_channel_names_by_pos(channels, channel_map),
        show_figure=False,
        figure_name='average_erp.pdf'
    )
    return model, figure_handles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    parser.add_argument('-p', '--parameters_file',
                        default='bcipy/parameters/parameters.json')
    args = parser.parse_args()

    log.info(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)
    model, data, labels, folder, fs, channels, channel_map = filter_inquiry_and_get_trials(args.data_folder, parameters)
    train(model, data, labels, folder, fs, channels, channel_map)
    log.info('Offline Analysis complete.')
