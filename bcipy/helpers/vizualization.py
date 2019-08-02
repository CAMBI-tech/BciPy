import os
import numpy as np
import matplotlib.pyplot as plt
from bcipy.helpers.load import load_csv_data, read_data_csv

import logging
log = logging.getLogger(__name__)


def generate_offline_analysis_screen(
        x,
        y,
        model=None,
        folder=None,
        plot_lik_dens=True,
        save_figure=True,
        down_sample_rate=2,
        fs=300,
        plot_x_ticks=8,
        plot_average=False,
        show_figure=False,
        channel_names=None) -> None:
    """ Offline Analysis Screen.

    Generates the information figure following the offlineAnalysis.
    The figure has multiple tabs containing the average ERP plots

    PARAMETERS
    ----------

    x(ndarray[float]): C x N x k data array
    y(ndarray[int]): N x k observation (class) array
        N is number of samples k is dimensionality of features
        C is number of channels
    model(): trained model for data
    folder(str): Folder of the data
    plot_lik_dens: boolean: whether or not to plot likelihood densities
    save_figures: boolean: whether or not to save the plots as PDF
    down_sample_rate: downsampling rate applied to signal (if any)
    fs (sampling_rate): original sampling rate of the signal
    plot_x_ticks: number of ticks desired on the ERP plot
    plot_average: boolean: whether or not to average over all channels
    show_figure: boolean: whether or not to show the figures generated
    channel_names: dict of channel names keyed by their position.
    """

    channel_names = channel_names or {}
    classes = np.unique(y)

    means = [np.squeeze(np.mean(x[:, np.where(y == i), :], 2))
             for i in classes]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    if plot_average:
        # find the mean across rows for non target and target
        non_target_mean = np.mean(means[0], axis=0)
        target_mean = np.mean(means[1], axis=0)

        # plot the means
        ax1.plot(non_target_mean)
        ax2.plot(target_mean)

    else:
        count = 0
        # Loop through all the channels and plot each on the non target/target
        # subplots
        while count < means[0].shape[0]:
            lbl = channel_names.get(count, count)
            ax1.plot(means[0][count, :], label=lbl)
            ax2.plot(means[1][count, :], label=lbl)
            count += 1
        ax1.legend(loc='upper left', prop={'size': 8})
        ax2.legend(loc='upper left', prop={'size': 8})

    # data points
    data_length = len(means[0][1, :])

    # generate appropriate data labels for the figure
    lower = 0

    # find the upper length of data and convert to seconds
    upper = data_length * down_sample_rate / fs * 1000

    # make the labels
    labels = [round(lower + x * (upper - lower) / (plot_x_ticks - 1)) for x in
              range(plot_x_ticks)]

    # make sure it starts at zero
    labels.insert(0, 0)

    # set the labels
    ax1.set_xticklabels(labels)
    ax2.set_xticklabels(labels)

    # Set common labels
    fig.text(0.5, 0.04, 'Time (Seconds)', ha='center', va='center')
    fig.text(0.06, 0.5, r'$\mu V$', ha='center', va='center',
             rotation='vertical')

    ax1.set_title('Non-target ERP')
    ax2.set_title('Target ERP')

    if save_figure:
        fig.savefig(
            os.path.join(
                folder,
                'mean_erp.pdf'),
            bbox_inches='tight',
            format='pdf')

    if plot_lik_dens:
        fig, ax = plt.subplots()
        x_plot = np.linspace(
            np.min(model.line_el[-2]), np.max(model.line_el[-2]), 1000)[:, np.newaxis]
        ax.plot(model.line_el[2][y == 0], -0.005 - 0.01 * np.random.random(
            model.line_el[2][y == 0].shape[0]), 'ro', label='class(-)')
        ax.plot(model.line_el[2][y == 1], -0.005 - 0.01 * np.random.random(
            model.line_el[2][y == 1].shape[0]), 'go', label='class(+)')
        for idx in range(len(model.pipeline[2].list_den_est)):
            log_dens = model.pipeline[2].list_den_est[idx].score_samples(
                x_plot)
            ax.plot(x_plot[:, 0], np.exp(log_dens), 'r-' *
                    (idx == 0) + 'g--' * (idx == 1), linewidth=2.0)

        ax.legend(loc='upper right')
        plt.title('Likelihoods Given the Labels')
        plt.ylabel('p(e|l)')
        plt.xlabel('scores')

        if save_figure:
            fig.savefig(
                os.path.join(
                    folder,
                    'lik_dens.pdf'),
                bbox_inches='tight',
                format='pdf')

    if show_figure:
        plt.show()


def visualize_csv_eeg_triggers(trigger_col=None):
    """Visualize CSV EEG Triggers.

    This function is used to load in CSV data and visualize device generated triggers.

    Input:
        trigger_col(int)(optional): Column location of triggers in csv file.
            It defaults to the last column.

    Output:
        Figure of Triggers
    """
    # Load in CSV
    file_name = load_csv_data()
    raw_data, stamp_time, channels, type_amp, fs = read_data_csv(file_name)

    # Pull out the triggers
    if not trigger_col:
        triggers = raw_data[-1]
    else:
        triggers = raw_data[trigger_col]

    # Plot the triggers
    plt.plot(triggers)

    # Add some titles and labels to the figure
    plt.title('Trigger Signal')
    plt.ylabel('Trigger Value')
    plt.xlabel('Samples')

    log.debug('Press Ctrl + C to exit!')
    # Show us the figure! Depending on your OS / IDE this may not close when
    #  The window is closed, see the message above
    plt.show()


if __name__ == '__main__':
    import pickle

    # load some x, y data from test files
    x = pickle.load(
        open(
            'bcipy/helpers/tests/resources/mock_x_generate_erp.pkl',
            'rb'))
    y = pickle.load(
        open(
            'bcipy/helpers/tests/resources/mock_y_generate_erp.pkl',
            'rb'))

    names = {
        0: 'P3',
        1: 'C3',
        2: 'F3',
        3: 'Fz',
        4: 'F4',
        5: 'C4',
        6: 'P4',
        7: 'Cz',
        8: 'A1',
        9: 'Fp1',
        10: 'Fp2',
        11: 'T3',
        12: 'T5',
        13: 'O1',
        14: 'O2',
        15: 'F7',
        16: 'F8',
        17: 'A2',
        18: 'T6',
        19: 'T4'
    }
    # generate the offline analysis screen. show figure at the end
    generate_offline_analysis_screen(
        x,
        y,
        folder='bcipy',
        plot_lik_dens=False,
        save_figure=False,
        show_figure=True,
        channel_names=names)
