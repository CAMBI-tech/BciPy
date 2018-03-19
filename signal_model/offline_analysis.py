from helpers.load import read_data_csv, load_experimental_data
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.train_model import train_pca_rda_kde_model
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.data_viz import generate_offline_analysis_screen
from helpers.triggers import trigger_decoder
import pickle
from time import time
from signal_model.offline_analysis_m import noise_data
import sys
import os


def offline_analysis(data_folder=None, add_artifacts=0):
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
        data_folder + '/raw_data.csv')

    t1 = time()
    # TODO: Read from parameters
    ds_rate = 2
    dat = sig_pro(raw_dat, fs=fs, k=ds_rate)

    # Process triggers.txt
    s_i, t_t_i, t_i, offset = trigger_decoder(mode=mode,
                                              trigger_loc=data_folder + '/triggers.txt')

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timestamp column.
    #                     CM            X3 X2           X1            TRG
    channel_map = [1]*8 + [0] + [1]*7 + [0]*2 + [1]*2 + [0] + [1]*3 + [0]

    x, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat,
                                      mode=mode, fs=fs, k=ds_rate, offset=offset,
                                      channel_map=channel_map)

    if add_artifacts:
        x = noise_data(x, y, 300, add_artifacts)

    model, auc_cv = train_pca_rda_kde_model(x, y, k_folds=10)
    t1 = time() - t1

    if not os.path.exists(data_folder+'/{}'.format(add_artifacts)):
        os.makedirs(data_folder+'/{}'.format(add_artifacts))

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder+'/{}'.format(add_artifacts))

    print('Saving the model!')
    with open(data_folder + '{}/model_duration_{}_auccv_{}.pkl'.format(add_artifacts, t1, auc_cv), 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == "__main__":
    try:
        ratio = sys.argv[1]
    except Exception as e:
        ratio = 10

    print 'Noisy sample rate: %{}'.format(ratio)

    #sample_calib_path = '/gss_gpfs_scratch/kadioglu.b/data/b/Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'
    sample_calib_path = None

    offline_analysis(data_folder=sample_calib_path, add_artifacts=ratio)
