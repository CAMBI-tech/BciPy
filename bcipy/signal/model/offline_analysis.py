import pickle
import logging

from bcipy.helpers.load import (
    read_data_csv,
    load_experimental_data,
    load_json_parameters)
from bcipy.signal.process.filter import bandpass, notch, downsample
from bcipy.signal.model.mach_learning.train_model import train_pca_rda_kde_model
from bcipy.helpers.task import trial_reshaper
from bcipy.helpers.vizualization import generate_offline_analysis_screen
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition import analysis_channels,\
    analysis_channel_names_by_pos
from bcipy.helpers.stimuli import play_sound

log = logging.getLogger(__name__)


def offline_analysis(data_folder: str = None,
                     parameters: dict = {}, alert_finished: bool = True):
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
        - generates and saves offline analysis screen
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

    raw_dat, _, channels, type_amp, fs = read_data_csv(
        data_folder + '/' + raw_data_file)

    log.info(f'Channels read from csv: {channels}')
    log.info(f'Device type: {type_amp}')

    # Remove 60hz noise with a notch filter
    notch_filter_data = notch.notch_filter(raw_dat, fs, frequency_to_remove=notch_filter)

    # bandpass filter
    filtered_data = bandpass.butter_bandpass_filter(
        notch_filter_data, lp_filter, hp_filter, fs, order=filter_order)

    # downsample
    data = downsample.downsample(filtered_data, factor=downsample_rate)

    # Process triggers.txt
    _, t_t_i, t_i, offset = trigger_decoder(
        mode=mode,
        trigger_path=f'{data_folder}/{triggers_file}')

    offset = offset + static_offset

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timespamp column.
    channel_map = analysis_channels(channels, type_amp)

    x, y, _, _ = trial_reshaper(t_t_i, t_i, data,
                                mode=mode, fs=fs, k=downsample_rate,
                                offset=offset,
                                channel_map=channel_map,
                                trial_length=trial_length)

    model, auc = train_pca_rda_kde_model(x, y, k_folds=k_folds)

    log.info('Saving offline analysis plots!')

    # After obtaining the model get the transformed data for plotting purposes
    model.transform(x)
    generate_offline_analysis_screen(
        x, y, model=model, folder=data_folder,
        down_sample_rate=downsample_rate,
        fs=fs, save_figure=True, show_figure=False,
        channel_names=analysis_channel_names_by_pos(channels, channel_map))

    log.info('Saving the model!')
    with open(data_folder + f'/model_{auc}.pkl', 'wb') as output:
        pickle.dump(model, output)

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

    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s')

    log.info(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)
    offline_analysis(args.data_folder, parameters)

    log.info('Offline Analysis complete.')
