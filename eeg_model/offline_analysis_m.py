# -*- coding: utf-8 -*-
import numpy as np
from helpers.load import read_data_csv, load_experimental_data
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model, train_m_estimator_pipeline
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.data_viz import generate_offline_analysis_screen
from helpers.triggers import trigger_decoder
import pickle
import matlab.engine

def load_data_matlab_csv(k=1):
    # This function is for loading data collected with matlab.
    eng = matlab.engine.start_matlab()
    # Below should take as parameter the path of your matlab code for RSVP Keyboard.
    eng.addpath(eng.genpath('C:\\Users\\Berkan\\Desktop\\GITProjects\\rsvp-keyboard'))
    x, y = eng.python_helper(k, nargout=2)

    return np.array(x), np.array(y)


def offline_analysis_m(data_folder=None, from_matlab=False, method='regular'):
    """ Train the model by first loading the data and save the trained model.
        Args:
            data_folder(str): Path to data folder
            from_matlab(boolean): If True use csv file from Matlab
        """

    k = 2

    # If data was recorded with Matlab
    if from_matlab:
        x, y = load_data_matlab_csv(k=k)
        y = np.squeeze(y)
        # indices = np.random.permutation(len(y))
        # x = x[:,indices,:]
        # y = y[indices]
    else:
        if not data_folder:
            data_folder = load_experimental_data()

        raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
            data_folder + '/raw_data.csv')

        dat = sig_pro(raw_dat, fs=fs, k=k)

        # Process triggers.txt
        s_i, t_t_i, t_i, offset = trigger_decoder(mode='calibration',
                                          trigger_loc=data_folder + '/triggers.txt')

        # Channel map can be checked from raw_data.csv file.
        # read_data_csv already removes the timespamp column.
        #                     CM            X3 X2           X1            TRG
        channel_map = [1] * 8 + [0] + [1] * 7 + [0] * 2 + [1] * 2 + [0] + [1] * 3 + [0]
        x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat, mode='calibration',
                                          fs=fs, k=k,
                                          channel_map=channel_map, offset=offset)

    if method=='regular':
        model = train_pca_rda_kde_model(x, y, k_folds=10)
    elif method=='m-estimator':
        model = train_m_estimator_pipeline(x, y, k_folds=10)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == '__main__':
    offline_analysis_m(from_matlab=False, data_folder='C:/Users/Berkan/Desktop/GITProjects/bci/data/andac/andac_Mon_22_Jan_2018_1236_Eastern Standard Time', method='m-estimator')


