import numpy as np
import matplotlib as mpl
from helpers.load import read_data_csv
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
import pickle
from helpers.load import load_experimental_data

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

    # TODO: Ask for a file location if not exists
    if not data_folder:
        data_folder = load_experimental_data()

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_folder + '/rawdata.csv')
    ds_rate = 2  # Read from parameters file the down-sampling rate
    dat = sig_pro(raw_dat, fs=fs, k=ds_rate)

    # Get data and labels
    x, y = trial_reshaper(data_folder + '/triggers.txt', dat, fs=fs,
                          k=ds_rate)

    # Determine on number of folds based on the data!
    k_folds = 4
    model = train_pca_rda_kde_model(x, y, k_folds=k_folds)

    fig, ax = plt.subplots()
    x_plot = np.linspace(np.min(model.line_el[-1]), np.max(model.line_el[-1]),
                         1000)[:, np.newaxis]
    ax.plot(model.line_el[2][y == 0], -0.005 - 0.01 * np.random.random(
        model.line_el[2][y == 0].shape[0]), 'ro', label='class(-)')
    ax.plot(model.line_el[2][y == 1], -0.005 - 0.01 * np.random.random(
        model.line_el[2][y == 1].shape[0]), 'go', label='class(+)')
    for idx in range(len(model.pipeline[2].list_den_est)):
        log_dens = model.pipeline[2].list_den_est[idx].score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens),
                'r-' * (idx == 0) + 'g--' * (idx == 1), linewidth=2.0)

    ax.legend(loc='upper right')
    plt.title('Likelihoods Given the Labels')
    plt.ylabel('p(e|l)')
    plt.xlabel('scores')
    plt.show()
    print('Saving the model!')
    with open(data_folder + '/model.pkl', 'wb') as output:
        pickle.dump(model, output)

    return 0


def main():
    offline_analysis()

    return 0


if __name__ == "__main__":
    main()
