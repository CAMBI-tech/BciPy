from helpers.load import read_data_csv, load_experimental_data
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.train_model import train_pca_rda_kde_model
from signal_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.triggers import trigger_decoder
from helpers.data_viz import generate_offline_analysis_screen
from time import time
import pickle


def offline_analysis(data_folder=None):
    """

    :param data_folder:
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

    model, auc = train_pca_rda_kde_model(x, y)

    t1 = time() - t1
    print 'Completed in {} mins'.format(t1/60.)

    print('Saving offline analysis plots!')
    generate_offline_analysis_screen(x, y, model, data_folder)

    print('Saving the model!')
    with open(data_folder + '/model_auc_%2.0f.pkl' % (auc*100), 'wb') as output:
        pickle.dump(model, output)
    return model


if __name__ == "__main__":

    calib_path = None
    offline_analysis(data_folder=calib_path)
