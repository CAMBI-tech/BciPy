from bcipy.helpers.load import read_data_csv, load_experimental_data, \
    load_json_parameters
from bcipy.signal_processing.sig_pro import sig_pro
from bcipy.signal_model.mach_learning.train_model import train_pca_rda_kde_model
from bcipy.helpers.bci_task_related import trial_reshaper
from bcipy.helpers.data_vizualization import generate_offline_analysis_screen
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition_related import analysis_channels
import pickle


def offline_analysis(data_folder=None, parameters={}):
    """ Gets calibration data and trains the model in an offline fashion.
        pickle dumps the model into a .pkl folder
        Args:
            data_folder(str): folder of the data
                save all information and load all from this folder

        Duty cycle
        - reads data and information from a .csv calibration file
        - reads trigger information from a .txt trigger file
        - filters data
        - reshapes and labels the data for the training procedure
        - fits the model to the data
            - uses cross validation to select parameters
            - based on the parameters, trains system using all the data
        - pickle dumps model into .pkl file
        - generates and saves offline analysis screen
    """

    if not data_folder:
        data_folder = load_experimental_data()

    mode = 'calibration'

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_folder + '/' + parameters.get('raw_data_name', 'raw_data.csv'))

    print(f'Channels read from csv: {channels}')
    print(f'Device type: {type_amp}')

    downsample_rate = parameters.get('down_sampling_rate', 2)
    filtered_data = sig_pro(raw_dat, fs=fs, k=downsample_rate)

    # Process triggers.txt
    triggers_file = parameters.get('triggers_file_name', 'triggers.txt')
    _, t_t_i, t_i, offset = trigger_decoder(
        mode=mode,
        trigger_path=f"{data_folder}/{triggers_file}")

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timespamp column.
    channel_map = analysis_channels(channels, type_amp)

    x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, filtered_data,
                                      mode=mode, fs=fs, k=downsample_rate,
                                      offset=offset,
                                      channel_map=channel_map,
                                      trial_length=parameters.get('collection_window_after_trial_length'))

    k_folds = parameters.get('k_folds', 10)
    model = train_pca_rda_kde_model(x, y, k_folds=10)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    parser.add_argument('-p', '--parameters_file',
                        default='bcipy/parameters/parameters.json')
    args = parser.parse_args()

    print(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)
    offline_analysis(args.data_folder, parameters)
