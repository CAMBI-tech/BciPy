# -*- coding: utf-8 -*-
from bcipy.helpers.load import load_experimental_data, read_data_csv, load_json_parameters
from bcipy.signal_processing.sig_pro import sig_pro
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.bci_task_related import trial_reshaper
from bcipy.signal_model.mach_learning.train_model import train_m_estimator_pipeline
from bcipy.helpers.data_vizualization import generate_offline_analysis_screen
from bcipy.helpers.acquisition_related import analysis_channels
from time import time
import pickle


def offline_analysis_m(data_folder=None, parameters={}):
    """
    Starts offline analysis on calibration data using robust method.
    :param data_folder: Directory of calibration file
    :return: model
    """

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
    print('Completed in {} mins'.format(t1/60.))

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/m_model_auc_%2.0f.pkl' % (model.last_cv_auc*100), 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == '__main__':
    # IMPORTANT: In calibration number of sequences should be higher than 120
    # SUGGESTED: (6 negative 1 positive) 150 sequences.
    import argparse
    print("IMPORTANT: In robust calibration number of sequences should be higher than 120\n"
          "SUGGESTED: (6 negative 1 positive) 150 sequences.")

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    parser.add_argument('-p', '--parameters_file',
                        default='bcipy/parameters/parameters.json')
    args = parser.parse_args()

    print(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)

    offline_analysis_m(data_folder=args.data_folder)
