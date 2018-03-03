# -*- coding: utf-8 -*-
from helpers.load import read_data_csv, load_experimental_data
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.train_model import train_pca_rda_kde_model, train_m_estimator_pipeline
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.data_viz import generate_offline_analysis_screen
from helpers.triggers import trigger_decoder
import pickle
from time import time


def offline_analysis_m(data_folder=None):
    """ Train the model by first loading the data and save the trained model.
        Args:
            data_folder(str): Path to data folder
            method(str): 'regular' pipeline or 'm-estimator' pipeline
        """

    k = 2
    if not data_folder:
        data_folder = load_experimental_data()

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(data_folder + '/raw_data.csv')

    dat = sig_pro(raw_dat, fs=fs, k=k)

    # Process triggers.txt
    s_i, t_t_i, t_i, offset = trigger_decoder(mode='calibration',
                                              trigger_loc=data_folder + '/triggers.txt')

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timestamp column.
    #                       CM            X3 X2           X1            TRG
    channel_map = [1] * 8 + [0] + [1] * 7 + [0] * 2 + [1] * 2 + [0] + [1] * 3 + [0]
    x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat, mode='calibration',
                                      fs=fs, k=k,
                                      channel_map=channel_map, offset=offset)

    model, auc_cv = train_m_estimator_pipeline(x, y, k_folds=10)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder+'/mpca', auc_cv)

    print('Saving the model!')
    with open(data_folder + 'mpca/model.pkl', 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == '__main__':
    t1 = time()
    offline_analysis_m(data_folder=None)
    print 'Elapsed time: {} mins'.format((time() - t1)/60.)
