# -*- coding: utf-8 -*-
from helpers.load import load_experimental_data, read_data_csv
from signal_processing.sig_pro import sig_pro
from helpers.triggers import trigger_decoder
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from signal_model.mach_learning.train_model import train_m_estimator_pipeline
from helpers.data_viz import generate_offline_analysis_screen
from time import time
import pickle


def offline_analysis_m(data_folder=None):
    """
    Starts offline analysis on calibration data using robust method.
    :param data_folder: Directory of calibration file
    :return: model
    """

    k = 2
    if not data_folder:
        data_folder = load_experimental_data()

    t1 = time()

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(data_folder + '/raw_data.csv')

    dat = sig_pro(raw_dat, fs=fs, k=k)

    # Process triggers.txt
    s_i, t_t_i, t_i, offset = trigger_decoder(mode='calibration',
                                              trigger_loc=data_folder + '/triggers.txt')

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timestamp column.
    #                       CM            X3 X2           X1            TRG
    channel_map = [1] * 8 + [0] + [1] * 7 + [0] * 2 + [1] * 2 + [0] + [1] * 3 + [0]
    x, y, _, _ = trial_reshaper(t_t_i, t_i, dat, mode='calibration',
                                      fs=fs, k=k,
                                      channel_map=channel_map, offset=offset)

    model, auc = train_m_estimator_pipeline(x, y)

    t1 = time() - t1
    print 'Completed in {} mins'.format(t1/60.)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/m_model_auc_%2.0f.pkl' % (auc*100), 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == '__main__':
    # IMPORTANT: In calibration number of sequences should be higher than 120
    # SUGGESTED: (6 negative 1 positive) 150 sequences.

    calib_path = None
    offline_analysis_m(data_folder=calib_path)