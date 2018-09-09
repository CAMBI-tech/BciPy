import numpy as np
import matplotlib.pyplot as plt
from bcipy.helpers.load import load_csv_data, read_data_csv

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


def generate_offline_analysis_screen(
        x,
        y,
        model=None,
        folder=None,
        plot_lik_dens=True,
        save_figure=True,
        down_sample_rate=2,
        fs=300,
        plot_x_ticks=8) -> None:
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
    classes = np.unique(y)

    means = [np.squeeze(np.mean(x[:, np.where(y == i), :], 2)) for i in classes]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    count = 1
    while count < means[0].shape[0]:
        ax1.plot(means[0][count, :])
        ax2.plot(means[1][count, :])
        count += 1

    # data points
    data_length = len(means[0][1, :])

    labels = [(x * down_sample_rate / fs) * 1000 for x in range(data_length)
              if (x% round(data_length/plot_x_ticks)) == 0]

    labels.insert(0, 0)

    ax1.set_xticklabels(labels)
    ax2.set_xticklabels(labels)

    # Set common labels
    fig.text(0.5, 0.04, 'Time (Seconds)', ha='center', va='center')
    fig.text(0.06, 0.5, '$\mu V$', ha='center', va='center',
             rotation='vertical')

    ax1.set_title('Mean non-target ERP')
    ax2.set_title('Mean target ERP')

    if save_figure:
        fig.savefig(f'{folder}\mean_erp.pdf', bbox_inches='tight', format='pdf')


    if plot_lik_dens:
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

        if save_figure:
            fig.savefig(f'{folder}\lik_dens.pdf', bbox_inches='tight', format='pdf')


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

    logging.debug('Press Ctrl + C to exit!')
    # Show us the figure! Depending on your OS / IDE this may not close when
    #  The window is closed, see the message above
    plt.show()


if __name__ == '__main__':
    import pickle

    # load some x, y data from test files
    x = pickle.load(open('bcipy/helpers/tests/resources/mock_x_generate_erp.pkl', 'rb'))
    y = pickle.load(open('bcipy/helpers/tests/resources/mock_y_generate_erp.pkl', 'rb'))

    generate_offline_analysis_screen(x, y, folder='bcipy', plot_lik_dens=False, save_figure=True)