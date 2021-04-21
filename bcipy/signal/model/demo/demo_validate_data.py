import matplotlib.pylab as plt
import numpy as np
from bcipy.signal.model.mach_learning.density_estimation import KernelDensityEstimate
import matplotlib as mpl
from scipy.stats import iqr
from bcipy.helpers.load import read_data_csv, load_experimental_data
from bcipy.signal.process.filter import bandpass
from bcipy.signal.model.mach_learning.train_model import train_pca_rda_kde_model
from bcipy.helpers.task import trial_reshaper
from bcipy.helpers.triggers import trigger_decoder

mpl.use('TkAgg')


def _demo_validate_data():
    dim_x = 75
    num_x_p = 500
    num_x_n = 500

    num_ch = 20

    x_p_train = np.asarray(
        [np.random.randn(num_x_p, dim_x) for i in
         range(num_ch)])
    x_n_train = np.array(
        [np.random.randn(num_x_p, dim_x) for i in
         range(num_ch)])
    y_p_train = [1] * num_x_p
    y_n_train = [0] * num_x_n

    x_train = np.concatenate((x_n_train, x_p_train), axis=1)
    y_train = np.concatenate((y_n_train, y_p_train), axis=0)

    permutation = np.random.permutation(x_train.shape[1])
    x_train = x_train[:, permutation, :]
    y_train = y_train[permutation]

    model, _ = train_pca_rda_kde_model(x_train, y_train, k_folds=10)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    x_plot = np.linspace(np.min(model.line_el[-1]), np.max(model.line_el[-1]),
                         1000)[:, np.newaxis]
    ax.plot(model.line_el[2][y_train == 0], -0.005 - 0.01 * np.random.random(
        model.line_el[2][y_train == 0].shape[0]), 'ro', label='class(-)')
    ax.plot(model.line_el[2][y_train == 1], -0.005 - 0.01 * np.random.random(
        model.line_el[2][y_train == 1].shape[0]), 'go', label='class(+)')
    for idx in range(len(model.pipeline[2].list_den_est)):
        log_dens = model.pipeline[2].list_den_est[idx].score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens),
                'r-' * (idx == 0) + 'g-' * (idx == 1), linewidth=2.0)

    ax.legend(loc='upper right')
    plt.title('Training Data')
    plt.ylabel('p(e|l)')
    plt.xlabel('scores')

    # Test
    x_p_test = np.asarray(
        [np.random.randn(num_x_p, dim_x) for i in
         range(num_ch)])
    x_n_test = np.array(
        [np.random.randn(num_x_p, dim_x) for i in range(num_ch)])
    y_p_test = [1] * num_x_p
    y_n_test = [0] * num_x_n

    x_test = np.concatenate((x_n_test, x_p_test), axis=1)
    y_test = np.concatenate((y_n_test, y_p_test), axis=0)

    permutation = np.random.permutation(x_test.shape[1])
    x_test = x_test[:, permutation, :]
    y_test = y_test[permutation]

    model.transform(x_test)

    ax.plot(model.line_el[2][y_test == 0], -0.01 - 0.01 * np.random.random(
        model.line_el[2][y_test == 0].shape[0]), 'bo', label='t_class(-)')
    ax.plot(model.line_el[2][y_test == 1], -0.01 - 0.01 * np.random.random(
        model.line_el[2][y_test == 1].shape[0]), 'ko', label='t_class(+)')

    bandwidth = 1.06 * min(np.std(model.line_el[2]),
                           iqr(model.line_el[2]) / 1.34) * np.power(
        model.line_el[2].shape[0], -0.2)
    test_kde = KernelDensityEstimate(bandwidth=bandwidth)
    test_kde.fit(model.line_el[2], y_test)

    for idx in range(len(model.pipeline[2].list_den_est)):
        log_dens = test_kde.list_den_est[idx].score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens),
                'b--' * (idx == 0) + 'k--' * (idx == 1), linewidth=2.0)

    ax.legend(loc='upper right')
    plt.title('Training Data')
    plt.ylabel('p(e|l)')
    plt.xlabel('scores')
    plt.show()


def _demo_validate_real_data():
    ds_rate = 2
    channel_map = [1] * 16 + [0, 0, 1, 1, 0, 1, 1, 1, 0]
    data_train_folder = load_experimental_data()

    mode = 'calibration'

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_train_folder + '/rawdata.csv')

    dat = bandpass.text_filter(raw_dat, fs=fs, k=ds_rate)

    # Get data and labels
    s_i, t_t_i, t_i = trigger_decoder(mode=mode,
                                      trigger_loc=data_train_folder + '/triggers.txt')
    x_train, y_train, num_inq, _ = trial_reshaper(t_t_i, t_i, dat, mode=mode,
                                                  fs=fs,
                                                  k=ds_rate,
                                                  channel_map=channel_map)

    model = train_pca_rda_kde_model(x_train, y_train, k_folds=10)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    x_plot = np.linspace(np.min(model.line_el[-1]), np.max(model.line_el[-1]),
                         1000)[:, np.newaxis]
    ax.plot(model.line_el[2][y_train == 0], -0.005 - 0.01 * np.random.random(
        model.line_el[2][y_train == 0].shape[0]), 'ro', label='class(-)')
    ax.plot(model.line_el[2][y_train == 1], -0.005 - 0.01 * np.random.random(
        model.line_el[2][y_train == 1].shape[0]), 'go', label='class(+)')
    for idx in range(len(model.pipeline[2].list_den_est)):
        log_dens = model.pipeline[2].list_den_est[idx].score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens),
                'r-' * (idx == 0) + 'g-' * (idx == 1), linewidth=2.0)

    ax.legend(loc='upper right')
    plt.title('Training Data')
    plt.ylabel('p(e|l)')
    plt.xlabel('scores')

    # Test
    data_test_folder = load_experimental_data()

    mode = 'calibration'

    raw_dat, stamp_time, channels, type_amp, fs = read_data_csv(
        data_test_folder + '/rawdata.csv')
    dat = bandpass.text_filter(raw_dat, fs=fs, k=ds_rate)

    # Get data and labels
    s_i, t_t_i, t_i = trigger_decoder(mode=mode,
                                      trigger_loc=data_test_folder + '/triggers.txt')
    x_test, y_test, num_inq, _ = trial_reshaper(t_t_i, t_i, dat, mode=mode,
                                                fs=fs,
                                                k=ds_rate,
                                                channel_map=channel_map)

    model.transform(x_test)

    ax.plot(model.line_el[2][y_test == 0], -0.01 - 0.01 * np.random.random(
        model.line_el[2][y_test == 0].shape[0]), 'bo', label='t_class(-)')
    ax.plot(model.line_el[2][y_test == 1], -0.01 - 0.01 * np.random.random(
        model.line_el[2][y_test == 1].shape[0]), 'ko', label='t_class(+)')

    bandwidth = 1.06 * min(np.std(model.line_el[2]),
                           iqr(model.line_el[2]) / 1.34) * np.power(
        model.line_el[2].shape[0], -0.2)
    test_kde = KernelDensityEstimate(bandwidth=bandwidth)
    test_kde.fit(model.line_el[2], y_test)

    for idx in range(len(model.pipeline[2].list_den_est)):
        log_dens = test_kde.list_den_est[idx].score_samples(x_plot)
        ax.plot(x_plot[:, 0], np.exp(log_dens),
                'b--' * (idx == 0) + 'k--' * (idx == 1), linewidth=2.0)

    ax.legend(loc='upper right')
    plt.title('Training Data')
    plt.ylabel('p(e|l)')
    plt.xlabel('scores')

    plt.show()


if __name__ == "__main__":
    _demo_validate_data()
