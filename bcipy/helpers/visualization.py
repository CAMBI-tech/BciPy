import logging
from os import path
from typing import List, Optional

from matplotlib.figure import Figure
from mne.io import read_raw_edf
import matplotlib.pyplot as plt
import numpy as np

from bcipy.helpers.load import choose_csv_file, load_raw_data

log = logging.getLogger(__name__)


def visualize_erp(
        data,
        labels,
        fs,
        class_labels: List[str] = ['Non-Target', 'Target'],
        plot_average: bool = False,
        show_figure: bool = False,
        save_path: Optional[str] = None,
        figure_name: str = 'mean_erp.pdf',
        channel_names: Optional[dict] = None) -> Figure:
    """ Visualize ERP.

    Generates a comparative ERP figure following a task execution. Given a set of trailed data,
    and labels describing two classes, they are plotted and may be saved or shown in a window.

    Returns a list of the figure handles created.

    PARAMETERS
    ----------
    N: samples
    k: features
    C: channels

    data(ndarray[float]): C x N x k data array
    labels(ndarray[int]): N x k observation (class) array. Assumed to be two classes [0, 1]
    fs (sampling_rate): sampling rate of the current data signal
    class_labels: list of legend names for the respective classes for plotting (0 ,1).
    plot_average: boolean: whether or not to average over all channels
    show_figure: boolean: whether or not to show the figures generated
    save_path: optional path to a save location of the figure generated
    figure_name: name of the figure to be used when save_path provided
    channel_names: dict of channel names keyed by their position.
    """
    channel_names = channel_names or {}
    classes = np.unique(labels)

    means = [np.squeeze(np.mean(data[:, np.where(labels == i), :], 2))
             for i in classes]

    data_length = len(means[0][1, :])

    # set upper and lower bounds of figure legend
    lower = 0
    upper = data_length / fs * 1000  # convert to seconds

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)

    if plot_average:
        # find the mean across rows for non target and target
        class_one_mean = np.mean(means[0], axis=0)
        class_two_mean = np.mean(means[1], axis=0)

        # plot the means
        ax1.plot(class_one_mean, label=class_labels[0])
        ax1.plot(class_two_mean, label=class_labels[1])
        ax1.legend(loc='upper left', prop={'size': 8})

    else:
        ax2 = fig.add_subplot(212)
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

    # make the labels
    figure_labels = np.linspace(lower, upper, num=8).astype(int).tolist()
    figure_labels.insert(0, 0)  # needed to force a 0 starting tick

    ax1.set_xticklabels(figure_labels)

    if not plot_average:
        ax2.set_xticklabels(figure_labels)
        ax1.set_title(class_labels[0])
        ax2.set_title(class_labels[1])
    else:
        ax1.set_title(f'Average {class_labels[0]} vs. {class_labels[1]}')

    # Set common figure labels
    fig.text(0.5, 0.04, 'Time (Seconds)', ha='center', va='center')
    fig.text(0.06, 0.5, r'$\mu V$', ha='center', va='center',
             rotation='vertical')

    if show_figure:
        plt.show()

    if save_path:
        fig.savefig(
            path.join(
                save_path,
                figure_name),
            bbox_inches='tight',
            format='pdf')

    return fig


def plot_edf(edf_path: str, auto_scale: bool = False):
    """Plot data from the raw edf file. Note: this works from an iPython
    session but seems to throw errors when provided in a script.

    Parameters
    ----------
        edf_path - full path to the generated edf file
        auto_scale - optional; if True will scale the EEG data; this is
            useful for fake (random) data but makes real data hard to read.
    """
    edf = read_raw_edf(edf_path, preload=True)
    if auto_scale:
        edf.plot(scalings='auto')
    else:
        edf.plot()


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
    data = load_raw_data(choose_csv_file())
    raw_data = data.by_channel()

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
    visualize_erp(
        x,
        y,
        150,
        # save_path='.', # uncomment to save to current working directory
        show_figure=True,
        plot_average=False,
        channel_names=names)
