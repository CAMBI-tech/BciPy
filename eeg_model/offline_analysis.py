import numpy as np
import matplotlib as mpl
from helpers.load import read_data_csv
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model
from scipy.io import loadmat
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
import pickle

mpl.use('TkAgg')
import matplotlib.pylab as plt


def offline_analysis(data_folder):
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

    # Ask for a file location if not exists

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_folder + '/rawdata.csv')
    ds_rate = 2  # Read from parameters file the down-sampling rate
    dat = sig_pro(raw_dat, fs=fs, k=ds_rate)

    # Get data and labels
    x, y = trial_reshaper(data_folder + '/display/triggers.txt', dat, fs=fs,
                          k=ds_rate)

    model = train_pca_rda_kde_model(x, y)
    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)

    return 0


def main():
    data_folder = 'C:/Users/Aziz/Desktop/GIT/bci'
    offline_analysis(data_folder)

    return 0


if __name__ == "__main__":
    main()
