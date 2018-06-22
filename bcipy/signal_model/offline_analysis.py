from bcipy.helpers.load import read_data_csv, load_experimental_data, \
    load_json_parameters
from bcipy.signal_processing.sig_pro import sig_pro
from bcipy.signal_model.mach_learning.train_model import train_pca_rda_kde_model, train_m_estimator_pipeline
from bcipy.helpers.bci_task_related import trial_reshaper
from bcipy.helpers.data_visualization import generate_offline_analysis_screen
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition_related import analysis_channels

import pickle
from time import time


def offline_analysis(data_folder=None, parameters={}, mode="regular"):
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

    if mode == "regular":
        if not data_folder:
            data_folder = load_experimental_data()

        mode = 'calibration'

        raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
            data_folder + '/' + parameters.get('raw_data_name', 'raw_data.csv'))

        print(f'Channels read from csv: {channels}')
        print(f'Device type: {type_amp}')

        downsample_rate = parameters.get('down_sampling_rate', 2)

        t1 = time()

        filtered_data = sig_pro(raw_dat, fs=fs, k=downsample_rate)

        # Process triggers.txt
        triggers_file = parameters.get('triggers_file_name', 'triggers.txt')
        _, t_t_i, t_i, offset = trigger_decoder(mode=mode,
                                                trigger_loc=f"{data_folder}/{triggers_file}")

        # Channel map can be checked from raw_data.csv file.
        # read_data_csv already removes the timespamp column.
        channel_map = analysis_channels(channels, type_amp)

        x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, filtered_data,
                                          mode=mode, fs=fs, k=downsample_rate,
                                          offset=offset,
                                          channel_map=channel_map)

        k_folds = parameters.get('k_folds', 10)
        model = train_pca_rda_kde_model(x, y, k_folds=10)

        t1 = time() - t1
        print('Completed in {} mins'.format(t1/60.))

        print('Saving offline analysis plots!')
        generate_offline_analysis_screen(x, y, model, data_folder)

        print('Saving the model!')
        with open(data_folder + '/model_auc_%2.0f.pkl' % (model.last_cv_auc*100), 'wb') as output:
            pickle.dump(model, output)
        return model

    elif mode == "robust":

        if not data_folder:
            data_folder = load_experimental_data()

        raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(data_folder + '/raw_data.csv')

        print(f'Channels read from csv: {channels}')
        print(f'Device type: {type_amp}')

        k = parameters.get('down_sampling_rate', 2)

        t1 = time()

        dat = sig_pro(raw_dat, fs=fs, k=k)

        triggers_file = parameters.get('triggers_file_name', 'triggers.txt')

        # Process triggers.txt
        _, trial_target_info, timing_info, offset = trigger_decoder(
            mode='calibration', trigger_loc=f"{data_folder}/{triggers_file}")

        # Channel map can be checked from raw_data.csv file.
        # read_data_csv already removes the timestamp column.
        channel_map = analysis_channels(channels, type_amp)
        x, y, _, _ = trial_reshaper(trial_target_info, timing_info, dat, mode='calibration',
                                    fs=fs, k=k,
                                    channel_map=channel_map, offset=offset)

        model = train_m_estimator_pipeline(x, y)

        t1 = time() - t1
        print('Completed in {} mins'.format(t1 / 60.))

        print('Saving offline analysis plots!')
        generate_offline_analysis_screen(x, y, model, data_folder)

        print('Saving the model!')
        with open(data_folder + '/m_model_auc_%2.0f.pkl' % (model.last_cv_auc * 100), 'wb') as output:
            pickle.dump(model, output)
        return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    parser.add_argument('-p', '--parameters_file',
                        default='bcipy/parameters/parameters.json')
    parser.add_argument('-m', '--mode', default='regular')

    args = parser.parse_args()

    if args.mode == 'robust':
        print('IMPORTANT: In robust calibration number of sequences should be higher than 120\n'
              'SUGGESTED: (6 negative 1 positive) 150 sequences.')

    print(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)

    offline_analysis(args.data_folder, parameters, mode=args.mode)
