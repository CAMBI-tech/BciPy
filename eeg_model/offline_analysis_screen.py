from helpers.load import read_data_csv, load_experimental_data
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.trial_reshaper import trial_reshaper


def generate_offline_analysis_screen(x, y, folder):
    """ Offline Analysis Screen.
    Generates the information figure following the offlineAnalysis.
    The figure has multiple tabs containing the average ERP plots

    Args:
        x(ndarray[float]): C x N x k data array
        y(ndarray[int]): N x k observation (class) array
            N is number of samples k is dimensionality of features
            C is number of channels
        folder(str): Folder of the data

        """
    # import
    import numpy as np
    import matplotlib.pyplot as plt

    classes = np.unique(y)
    means = [np.mean(x[:, :, np.where(y == i)], 3) for i in classes]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    count = 1
    while count < np.shape(means[1])[1]:
        ax1.plot(means[0][:, count])
        ax1.plot(means[0][:, count])
        ax2.plot(means[1][:, count])
        ax2.plot(means[1][:, count])
        count += 1

    # Set common labels
    fig.text(0.5, 0.04, 'time Samples [n]', ha='center', va='center')
    fig.text(0.06, 0.5, '$\mu V$', ha='center', va='center',
             rotation='vertical')

    ax1.set_title(
        'Mean distractor ERP (averaged over trials in the calibration data')
    ax2.set_title(
        'Mean target ERP (averaged over trials in the calibration data)')

    fig.savefig(folder + "\mean_erp.pdf", bbox_inches='tight', format='pdf')


def _demo_gen_offline_analysis_screen():
    data_folder = load_experimental_data()
    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_folder + '/rawdata.csv')
    ds_rate = 2  # Read from parameters file the down-sampling rate
    dat = sig_pro(raw_dat, fs=fs, k=ds_rate)

    # Get data and labels
    x, y = trial_reshaper(data_folder + '/triggers.txt', dat, fs=fs,
                          k=ds_rate)

    generate_offline_analysis_screen(x, y, data_folder)


def main():
    _demo_gen_offline_analysis_screen()

    return 0


if __name__ == "__main__":
    main()
