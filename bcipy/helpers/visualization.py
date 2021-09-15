import os
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from bcipy.helpers.load import choose_csv_file, load_raw_data
from mne.io import read_raw_edf
from typing import List

import logging

log = logging.getLogger(__name__)


def generate_offline_analysis_screen(
    x,
    y,
    model=None,
    folder=None,
    save_figure=True,
    down_sample_rate=2,
    fs=300,
    plot_x_ticks=8,
    plot_average=False,
    show_figure=False,
    channel_names=None,
) -> List[Figure]:
    """Offline Analysis Screen.

    Generates the information figure following the offlineAnalysis.
    The figure has multiple tabs containing the average ERP plots.

    Returns a list of the figure handles created.

    PARAMETERS
    ----------

    x(ndarray[float]): C x N x k data array
    y(ndarray[int]): N x k observation (class) array
        N is number of samples k is dimensionality of features
        C is number of channels
    model(): trained model for data
    folder(str): Folder of the data
    save_figures: boolean: whether or not to save the plots as PDF
    down_sample_rate: downsampling rate applied to signal (if any)
    fs (sampling_rate): original sampling rate of the signal
    plot_x_ticks: number of ticks desired on the ERP plot
    plot_average: boolean: whether or not to average over all channels
    show_figure: boolean: whether or not to show the figures generated
    channel_names: dict of channel names keyed by their position.
    """
    fig_handles = []

    channel_names = channel_names or {}
    classes = np.unique(y)

    means = [np.squeeze(np.mean(x[:, np.where(y == i), :], 2)) for i in classes]

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
        ax1.legend(loc="upper left", prop={"size": 8})
        ax2.legend(loc="upper left", prop={"size": 8})

    # data points
    data_length = len(means[0][1, :])

    # generate appropriate data labels for the figure
    lower = 0

    # find the upper length of data and convert to seconds
    upper = data_length * fs * 1000

    # make the labels
    labels = [round(lower + x * (upper - lower) / (plot_x_ticks - 1)) for x in range(plot_x_ticks)]

    # make sure it starts at zero
    labels.insert(0, 0)

    # set the labels
    ax1.set_xticklabels(labels)
    ax2.set_xticklabels(labels)

    # Set common labels
    fig.text(0.5, 0.04, "Time (Seconds)", ha="center", va="center")
    fig.text(0.06, 0.5, r"$\mu V$", ha="center", va="center", rotation="vertical")

    ax1.set_title("Non-target ERP")
    ax2.set_title("Target ERP")

    if save_figure:
        fig.savefig(os.path.join(folder, "mean_erp.pdf"), bbox_inches="tight", format="pdf")

    fig_handles.append(fig)

    if show_figure:
        plt.show()

    return fig_handles


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
        edf.plot(scalings="auto")
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
    plt.title("Trigger Signal")
    plt.ylabel("Trigger Value")
    plt.xlabel("Samples")

    log.debug("Press Ctrl + C to exit!")
    # Show us the figure! Depending on your OS / IDE this may not close when
    #  The window is closed, see the message above
    plt.show()


if __name__ == "__main__":
    import pickle

    # load some x, y data from test files
    x = pickle.load(open("bcipy/helpers/tests/resources/mock_x_generate_erp.pkl", "rb"))
    y = pickle.load(open("bcipy/helpers/tests/resources/mock_y_generate_erp.pkl", "rb"))

    names = {
        0: "P3",
        1: "C3",
        2: "F3",
        3: "Fz",
        4: "F4",
        5: "C4",
        6: "P4",
        7: "Cz",
        8: "A1",
        9: "Fp1",
        10: "Fp2",
        11: "T3",
        12: "T5",
        13: "O1",
        14: "O2",
        15: "F7",
        16: "F8",
        17: "A2",
        18: "T6",
        19: "T4",
    }
    # generate the offline analysis screen. show figure at the end
    generate_offline_analysis_screen(x, y, folder="bcipy", save_figure=False, show_figure=True, channel_names=names)
