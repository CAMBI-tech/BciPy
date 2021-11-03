import logging
from pathlib import Path
from bcipy.helpers.acquisition import analysis_channel_names_by_pos, analysis_channels
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.stimuli import play_sound
from bcipy.helpers.system_utils import report_execution_time
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.process import get_default_transform


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s')


@report_execution_time
def offline_analysis(data_folder: str = None,
                     parameters: dict = {}, alert_finished: bool = True) -> SignalModel:
    """ Gets calibration data and trains the model in an offline fashion.
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

    mode = 'calibration'

    # extract relevant session information from parameters file
    trial_length = parameters.get('trial_length')
    triggers_file = parameters.get('trigger_file_name', 'triggers.txt')
    raw_data_file = parameters.get('raw_data_name', 'raw_data.csv')

    # get signal filtering information
    downsample_rate = parameters.get('down_sampling_rate', 2)
    notch_filter = parameters.get('notch_filter_frequency', 60)
    hp_filter = parameters.get('filter_high', 45)
    lp_filter = parameters.get('filter_low', 2)
    filter_order = parameters.get('filter_order', 2)

    # get offset and k folds
    static_offset = parameters.get('static_trigger_offset', 0)
    k_folds = parameters.get('k_folds', 10)

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    log.info(f'Channels read from csv: {channels}')
    log.info(f'Device type: {type_amp}')

    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    data, fs = default_transform(raw_data.by_channel(), fs)

    # Process triggers.txt
    _, t_t_i, t_i, offset = trigger_decoder(
        mode=mode,
        trigger_path=f'{data_folder}/{triggers_file}')

    offset = offset + static_offset

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    channel_map = analysis_channels(channels, type_amp)

    model = PcaRdaKdeModel(k_folds=k_folds)
    data, labels = model.reshaper(
        trial_labels=t_t_i,
        timing_info=t_i,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=parameters.get('stim_length'),
        offset=offset,
        channel_map=channel_map,
        trial_length=trial_length)

    log.info('Training model. This will take some time...')
    model.fit(data, labels)
    model_performance = model.evaluate(data, labels)

    log.info(f'Training complete [AUC={model_performance.auc:0.4f}]. Saving data...')

    model.save(data_folder + f'/model_{model_performance.auc:0.4f}.pkl')
    visualize_erp(
        data,
        labels,
        fs,
        save_path=data_folder,
        channel_names=analysis_channel_names_by_pos(channels, channel_map),
        show_figure=False
    )
    visualize_erp(
        data,
        labels,
        fs,
        plot_average=True,
        save_path=data_folder,
        channel_names=analysis_channel_names_by_pos(channels, channel_map),
        show_figure=False,
        figure_name='average_erp.pdf'
    )
    if alert_finished:
        offline_analysis_tone = parameters.get('offline_analysis_tone')
        play_sound(offline_analysis_tone)

    return model


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
    offline_analysis(args.data_folder, parameters)
    log.info('Offline Analysis complete.')
