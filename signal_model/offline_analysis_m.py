# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from helpers.load import load_experimental_data, read_data_csv
from signal_processing.sig_pro import sig_pro
from helpers.triggers import trigger_decoder
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from signal_model.mach_learning.train_model import train_m_estimator_pipeline

import numpy as np
import sklearn as sk
import random
import pickle
from time import time



def noise_data(dat, amplitude, length, p, channel_map):
    """

    :param dat:
    :param amplitude:
    :param length: number of samples of artifact, should be less than 75
    :param p:
    :return:
    """

    C, d = dat.shape

    if p > 0:
        print 'p>0'

    if p == .001:
        print 'p==.001'

    activations = np.random.binomial(1, p, d)

    # Below is for having one activation per length samples
    end = -99999
    for z in range(d):
        if activations[z] == 1:
            if z < end:
                activations[z] = 0
            else:
                end = z + length

    activations[d-length:] = 0

    artifact_start_indices = np.where(activations)[0]

    noise_type = 'artifact'

    if noise_type == 'gaussian':
        mean = amplitude*np.ones(length)

        for indice in artifact_start_indices:
            cov = 100 * sk.datasets.make_spd_matrix(length, random_state=15)
            for c in range(C):
                dat[c, indice:indice+length] += \
                    np.random.multivariate_normal(mean=mean, cov=cov)*np.hamming(length)

    elif noise_type == 'artifact':

        with open('C:/Users/Berkan/Desktop/data/jaw_muscle.pkl') as f:
        # with open('/gss_gpfs_scratch/kadioglu.b/data/jaw_muscle.pkl') as f:
            artifacts = pickle.load(f)  # CxN'xd
        N_prime = artifacts.shape[1]
        list_indexes = range(N_prime)

        for indice in artifact_start_indices:
            dat[np.where(channel_map)[0], indice:indice+length] += \
                amplitude*np.squeeze(artifacts[:, random.sample(list_indexes, 1), 0:length]*np.hamming(length))

    return dat


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

    t1 = time()

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
        dat_artifact = noise_data(dat=dat, amplitude=20, length=30, p=add_artifacts, channel_map=channel_map)
        x_artifact, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat_artifact, mode='calibration',
                                          fs=fs, k=k,
                                          channel_map=channel_map, offset=offset)

    model, auc_cv = train_m_estimator_pipeline(x, x_artifact, y, k_folds=10)

    t1 = time() - t1
    print 'Completed in {} mins'.format(t1/60.)


if __name__ == '__main__':
    try:
        percent_rate = np.float(sys.argv[1])
    except Exception as e:
        percent_rate = np.float('.000000001')
        print 'sys.argv failed'

    print 'Noise activation rate: {}'.format(percent_rate)
    np.random.seed(150)
    # sample_calib_path = '/gss_gpfs_scratch/kadioglu.b/data/b/Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'
    sample_calib_path = None

    offline_analysis_m(data_folder=sample_calib_path, add_artifacts=percent_rate)

