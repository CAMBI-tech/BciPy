# -*- coding: utf-8 -*-
from helpers.load import read_data_csv, load_experimental_data
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.train_model import train_m_estimator_pipeline
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.data_viz import generate_offline_analysis_screen
from helpers.triggers import trigger_decoder
import numpy as np
import random
import pickle
from time import time
import os
import sys


def noise_data(x, y, amplitude=1, ratio=5.):
    '''
    with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([obj0, obj1, obj2], f)

    with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
        obj0, obj1, obj2 = pickle.load(f)
    '''

    C, N, d = x.shape
    length = int(d/2)

    num_p = np.sum(y)*ratio/100
    num_n = (y.size - np.sum(y))*ratio/100

    index_p = np.where(y)[0]
    index_n = np.where(np.abs(y-1))[0]

    selected_p = random.sample(index_p, int(num_p))
    selected_n = random.sample(index_n, int(num_n))

    for p_i in selected_p:
        start_of_artifact = np.random.randint(low=0, high=d-length, size=1)[0]
        for c in range(C):
            x[c, p_i, start_of_artifact:start_of_artifact+length] += np.transpose(np.random.multivariate_normal(amplitude*np.ones(length),
                                                                                                   3000*np.eye(length, length)))

    for n_i in selected_n:
        start_of_artifact = np.random.randint(low=0, high=d-length, size=1)[0]
        for c in range(C):
            x[c, n_i, start_of_artifact:start_of_artifact+length] += np.transpose(np.random.multivariate_normal(amplitude*np.ones(length),
                                                                                                   3000*np.eye(length, length)))
    return x


def offline_analysis_m(data_folder=None, add_artifacts = 0):
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

    if add_artifacts:
        x = noise_data(x, y, amplitude=300, ratio=add_artifacts)

    model, auc_cv = train_m_estimator_pipeline(x, y, k_folds=10)

    if not os.path.exists(data_folder+'/mpca'):
        os.makedirs(data_folder+'/mpca')

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder+'/mpca', auc_cv)

    print('Saving the model!')
    with open(data_folder + 'mpca_model.pkl', 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == '__main__':
    try:
        ratio = sys.argv[1]
    except Exception as e:
        ratio = 10

    t1 = time()
    sample_calib_path = 'C:/Users/Berkan/Desktop/data/bci_main_demo_user/Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'

    offline_analysis_m(data_folder=sample_calib_path, add_artifacts=ratio)
    print 'Elapsed time: {} mins'.format((time() - t1)/60.)
