# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')

from helpers.load import load_experimental_data, read_data_csv
from signal_processing.sig_pro import sig_pro
from helpers.triggers import trigger_decoder
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from signal_model.mach_learning.train_model import train_m_estimator_pipeline

import numpy as np
import sklearn as sk
import pickle
from time import time

def noise_data(dat, amplitude, length, p, channel_map):
    """
        Add pre-recorded artifacts to data for degrading performance.
    :param dat:
    :param amplitude:
    :param length: number of samples of artifact, should be less than 75
    :param p:
    :return:
    """

    C, d = dat.shape

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
    print('Artifact start indices (#Artf. = {}):').format(len(artifact_start_indices))
    print(artifact_start_indices[0:14])

    noise_type = 'artifact'

    if noise_type == 'gaussian':
        mean = amplitude*np.ones(length)

        for indice in artifact_start_indices:
            cov = 100 * sk.datasets.make_spd_matrix(length, random_state=15)
            for c in range(C):
                dat[c, indice:indice+length] += \
                    np.random.multivariate_normal(mean=mean, cov=cov)*np.hamming(length)

    elif noise_type == 'artifact':

        # with open('C:/Users/Berkan/Desktop/data/jaw_muscle.pkl') as f:
        with open('/gss_gpfs_scratch/kadioglu.b/data/jaw_muscle.pkl') as f:
            artifacts = pickle.load(f)  # CxN'xd
        N_prime = artifacts.shape[1]
        list_indexes = range(N_prime)
        list_artifact_indices = np.random.choice(list_indexes, len(artifact_start_indices))
        print 'Chosen artifacts for chosen start indices:'
        print list_artifact_indices[0:14]

        for z in range(len(artifact_start_indices)):
            dat[np.where(channel_map)[0], artifact_start_indices[z]:artifact_start_indices[z]+length] += \
                amplitude*np.squeeze(artifacts[:, list_artifact_indices[z], 0:length]*np.hamming(length))

    return dat


def offline_analysis_m(data_folder=None, add_artifacts=0., leng=1, amp=1):
    """

    :param data_folder:
    :param add_artifacts:
    :param leng:
    :param amp:
    :return:
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

    # Adding pre-recorded artifacts below:
    if add_artifacts:
        dat_artifact = noise_data(dat=dat, amplitude=amp, length=leng, p=add_artifacts, channel_map=channel_map)

        x_artifact, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat_artifact, mode='calibration',
                                          fs=fs, k=k,
                                          channel_map=channel_map, offset=offset)

    model, auc_cv = train_m_estimator_pipeline(x, x_artifact, y, k_folds=10)

    t1 = time() - t1
    print 'Completed in {} mins'.format(t1/60.)


if __name__ == '__main__':
    try:
        act_rate = np.float(sys.argv[1])
        leng = int(np.float(sys.argv[2]))
        amp = np.float(sys.argv[3])
        seed = int(np.float(sys.argv[4]))
    except Exception as e:
        act_rate = .01
        leng = 30
        amp = 1
        seed = 7

    print 'Noise activation rate: {}'.format(act_rate)
    print 'Length: {}'.format(leng)
    print 'Amplitude: {}'.format(amp)
    print 'Random seed: {}\n'.format(seed)

    np.random.seed(seed)
    sample_calib_path = \
        '/gss_gpfs_scratch/kadioglu.b/data/Berkan_calib/Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'
    # sample_calib_path = 'C:\Users\Berkan\Desktop\data\Berkan_calib\Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'
    # sample_calib_path = None

    offline_analysis_m(data_folder=sample_calib_path, add_artifacts=act_rate, leng=leng, amp=amp)

