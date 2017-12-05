from helpers.load import read_data_csv, load_experimental_data
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.trial_reshaper import trial_reshaper


def generate_offline_analysis_screen(x, y, model, folder):
    """ Offline Analysis Screen.
    Generates the information figure following the offlineAnalysis.
    The figure has multiple tabs containing the average ERP plots

    Args:
        x(ndarray[float]): C x N x k data array
        y(ndarray[int]): N x k observation (class) array
            N is number of samples k is dimensionality of features
            C is number of channels
        model(): trained model for data
        folder(str): Folder of the data

        """
    # import
    import numpy as np
    import matplotlib.pyplot as plt

    classes = np.unique(y)
    means = [np.squeeze(np.mean(x[:, np.where(y == i), :], 2))
             for i in classes]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    count = 1
    while count < means[0].shape[0]:
        ax1.plot(means[0][count, :])
        ax2.plot(means[1][count, :])
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
    fig.savefig(folder + "\lik_dens.pdf", bbox_inches='tight', format='pdf')
