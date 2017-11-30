import numpy as np
import matplotlib as mpl
from helpers.load import read_data_csv, load_experimental_data
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from eeg_model.offline_analysis_screen import generate_offline_analysis_screen
from helpers.trigger_helpers import trigger_decoder
import pickle

mpl.use('TkAgg')
import matplotlib.pylab as plt


def offline_analysis(data_folder=None):
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
        data_folder + '/rawdata.csv')
    ds_rate = 2  # Read from parameters file the down-sampling rate
    dat = sig_pro(raw_dat, fs=fs, k=ds_rate)

    # Get data and labels
    s_i, t_t_i, t_i = trigger_decoder(mode=mode,
                                      trigger_loc=data_folder + '/triggers.txt')
    x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat, mode=mode, fs=fs,
                                      k=ds_rate)

    # Determine on number of folds based on the data!
    model = train_pca_rda_kde_model(x, y, k_folds=num_seq/10)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)
    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)

    return 0


def main():
    offline_analysis()

    return 0


if __name__ == "__main__":
    main()
