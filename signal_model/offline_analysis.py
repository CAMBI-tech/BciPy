import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from helpers.load import read_data_csv, load_experimental_data
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.train_model import train_pca_rda_kde_model
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from signal_model.offline_analysis_m import noise_data
from helpers.triggers import trigger_decoder
import numpy as np
from time import time


def offline_analysis(data_folder=None, add_artifacts=0., leng=1, amp=1):
    """

    :param data_folder:
    :param add_artifacts:
    :param leng:
    :return:
    """

    if not data_folder:
        data_folder = load_experimental_data()

    mode = 'calibration'

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_folder + '/raw_data.csv')

    t1 = time()
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
        dat_artifact = noise_data(dat=dat, amplitude=amp, length=leng, p=add_artifacts, channel_map=channel_map)
        x_artifact, y, num_seq, _ = trial_reshaper(t_t_i, t_i, dat_artifact,
                                          mode=mode, fs=fs, k=ds_rate, offset=offset,
                                          channel_map=channel_map)

    model, auc_cv = train_pca_rda_kde_model(x, x_artifact, y, k_folds=10)

    t1 = time() - t1
    print 'Completed in {} mins'.format(t1/60.)


if __name__ == "__main__":
    try:
        rate = np.float(sys.argv[1])
        leng = np.float(sys.argv[2])
        amp = np.float(sys.argv[3])
        seed = np.float(sys.argv[4])
    except Exception as e:
        rate = .01
        leng = 30
        amp = 1
        seed = 7

    print 'Noise activation rate: {}'.format(rate)
    print 'Length: {}'.format(leng)
    print 'Amplitude: {}'.format(amp)
    print 'Random seed: {}\n'.format(seed)

    np.random.seed(seed)
    sample_calib_path = \
        '/gss_gpfs_scratch/kadioglu.b/data/Berkan_calib/Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'
    # sample_calib_path = 'C:\Users\Berkan\Desktop\data\Berkan_calib\Berkan_Wed_28_Feb_2018_0209_Eastern Standard Time'
    # sample_calib_path = None

    offline_analysis(data_folder=sample_calib_path, add_artifacts=rate, leng=leng, amp=amp)
