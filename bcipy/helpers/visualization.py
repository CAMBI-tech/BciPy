# mypy: disable-error-code="attr-defined,union-attr,arg-type"
# needed for the ERPTransformParams
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from mne import Epochs
from scipy import linalg

import bcipy.acquisition.devices as devices
from bcipy.config import (DEFAULT_DEVICE_SPEC_FILENAME,
                          DEFAULT_GAZE_IMAGE_PATH, RAW_DATA_FILENAME,
                          TRIGGER_FILENAME, SESSION_LOG_FILENAME,
                          DEFAULT_PARAMETERS_PATH)
from bcipy.helpers.acquisition import analysis_channels
from bcipy.io.convert import convert_to_mne
from bcipy.io.load import choose_csv_file, load_raw_data, load_json_parameters
from bcipy.data.parameters import Parameters
from bcipy.data.raw_data import RawData
from bcipy.data.stimuli import mne_epochs
from bcipy.data.triggers import TriggerType, trigger_decoder
from bcipy.signal.process import (Composition, ERPTransformParams,
                                  get_default_transform)

logger = logging.getLogger(SESSION_LOG_FILENAME)


def clip_to_display(data, screen_limits):
    """Clip to Display. This function is used to clip the eye data to the display size. Removes
    > 1 values and < 0 values. Removes nan values.

    Input:
        data (np.ndarray): Eye data
        screen_limits (tuple): Screen limits

    Output:
        clipped_data (np.ndarray): Clipped eye data
    """
    clipped_data = np.clip(data, screen_limits[0], screen_limits[1])
    clipped_data = clipped_data[~np.isnan(clipped_data)]

    return clipped_data


def visualize_erp(
        raw_data: RawData,
        channel_map: List[int],
        trigger_timing: List[float],
        trigger_labels: List[int],
        trial_window: Tuple[float, float],
        transform: Optional[Composition] = None,
        plot_average: Optional[bool] = True,
        plot_topomaps: Optional[bool] = True,
        show: Optional[bool] = False,
        save_path: Optional[str] = None) -> List[Figure]:
    """ Visualize ERP.

    Generates a comparative ERP figure following a task execution. Given a set of trailed data,
    and labels describing two classes (Nontarget=0 and Target=1), they are plotted and may be saved
    or shown in a window.

    Returns a list of the figure handles created.

    PARAMETERS
    ----------
    raw_data(RawData): BciPy RawData
    channel_map(List[int]): Map of channels to remove (0) or keep (1)
    trigger_timing: list of trigger timing
    trigger_labels: list of trigger labels
    trial_window: Tuple[float, float]: time window to plot
    transform(Composition): Optional BciPy composition to apply to data before visualization
    plot_average: Optional[boolean]: whether or not to average over all channels. Default: True
    plot_topomap: Optional[boolean]: whether or not to plot all channels across target/nontarget. Default: True
    show: Optional[boolean]: whether or not to show the figures generated. Default: False
    save_path: optional path to a save location of the figure generated
    """
    # check for trial length in seconds from stimuli start (0.0)
    trial_length = trial_window[1] - 0.0

    # check for a baseline interval or set to None
    if trial_window[0] <= 0:
        baseline = (trial_window[0], 0.0)
    else:
        baseline = None

    mne_data = convert_to_mne(raw_data, channel_map=channel_map, transform=transform)
    epochs = mne_epochs(mne_data, trial_length, trigger_timing, trigger_labels, baseline=baseline)
    # *Note* We assume, as described above, two trigger classes are defined for use in trigger_labels
    # (Nontarget=0 and Target=1). This will map into two corresponding MNE epochs whose indexing starts at 1.
    # Therefore, epochs['1'] == Nontarget and epochs['2'] == Target.
    epochs = (epochs['1'], epochs['2'])
    figs = []
    if plot_average:
        figs.extend(visualize_evokeds(epochs, save_path=save_path, show=show))
    if plot_topomaps:
        # make a list of equally spaced times to plot topomaps using the time window
        # defined in the task parameters
        times = [round(trial_window[0] + i * (trial_window[1] - trial_window[0]) / 5, 1) for i in range(7)]

        # clip any times that are out of bounds of the time window or zero
        times = [time for time in times if trial_window[0] <= time <= trial_window[1] and time != 0]

        figs.extend(visualize_joint_average(
            epochs, ['Non-Target', 'Target'],
            save_path=save_path,
            show=show,
            plot_joint_times=times))

    return figs


def visualize_gaze(
        data: RawData,
        left_keys: List[str] = ['left_x', 'left_y'],
        right_keys: List[str] = ['right_x', 'right_y'],
        save_path: Optional[str] = None,
        show: Optional[bool] = False,
        img_path: Optional[str] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        heatmap: Optional[bool] = False,
        raw_plot: Optional[bool] = False) -> Figure:
    """Visualize Eye Data.

    Assumes that the data is collected using BciPy and a Tobii-nano eye tracker. The default
    image used is for the matrix calibration task on a 1920x1080 screen.

    Generates a comparative matrix figure following the execution of tasks. Given a set of
    trailed data, the gaze distribution are plotted and may be saved or shown in a window.

    Returns a list of the figure handles created.

    Parameters
    ----------
    data: RawData: Gaze RawData
    left_keys: Optional[List[str]]: List of channels for the left eye data. Default: ['left_x', 'left_y']
    right_keys: Optional[List[str]]: List of channels for the right eye data. Default: ['right_x', 'right_y']
    save_path: Optional[str]: optional path to a save location of the figure generated
    show: Optional[bool]: whether or not to show the figures generated. Default: False
    img_path: Optional[str]: Image to be used as the background. Default: matrix.png
    screen_size: Optional[Tuple[int, int]]: Size of the screen used for Calibration/Copy
        Phrase tasks.
        Default: (1920, 1080)
    heatmap: Optional[bool]: Whether or not to plot the heatmap. Default: False
    raw_plot: Optional[bool]: Whether or not to plot the raw gaze data. Default: False
    """

    title = f'{data.daq_type} '
    if heatmap:
        title += 'Heatmap '
    if raw_plot:
        title += 'Raw Gaze '

    if img_path is None:
        img_path = f'{DEFAULT_GAZE_IMAGE_PATH}'

    img = plt.imread(img_path)
    channels = data.channels
    left_eye_channel_map = [1 if channel in left_keys else 0 for channel in channels]
    left_eye_data, _, _ = data.by_channel_map(left_eye_channel_map)
    left_eye_x = left_eye_data[0]
    left_eye_y = left_eye_data[1]

    right_eye_channel_map = [1 if channel in right_keys else 0 for channel in channels]
    right_eye_data, _, _ = data.by_channel_map(right_eye_channel_map)
    right_eye_x = right_eye_data[0]
    right_eye_y = right_eye_data[1]
    eye_data = (left_eye_x, left_eye_y, right_eye_x, right_eye_y)

    # Transform the eye data to fit the display. Remove values < 1 & > 0, remove nan values
    screen_limits = (0, 1)
    clipped_data = []
    for i in eye_data:
        clipped_data.append(clip_to_display(i, screen_limits))

    lx, ly, rx, ry = clipped_data

    # Flip y axis
    ly = 1 - ly
    ry = 1 - ry

    # scale the eye data to the image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])

    if heatmap:
        # create a dataframe making a column for each x, y pair for both eyes and a column for the eye (left or right)
        df = pd.DataFrame({
            'x': np.concatenate((lx, rx)),
            'y': np.concatenate((ly, ry)),
            'eye': ['left'] * len(lx) + ['right'] * len(rx)
        })
        ax = sns.kdeplot(
            data=df,
            hue='eye',
            x='x',
            y='y',
            fill=False,
            thresh=0.05,
            levels=10,
            cmap="mako",
            colorbar=True)

    if raw_plot:
        # ax.scatter(lx, range(len(lx)), c='r', s=1)
        ax.scatter(lx, ly, c='r', s=1)
        ax.scatter(rx, ry, c='b', s=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(f'{title}Plot')

    if save_path is not None:
        plt.savefig(f"{save_path}/{title.lower().replace(' ', '_')}plot.png", dpi=fig.dpi)

    if show:
        plt.show()

    return fig


def visualize_gaze_inquiries(
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        means: Optional[np.ndarray] = None,
        covs: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = False,
        img_path: Optional[str] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        heatmap: Optional[bool] = False,
        raw_plot: Optional[bool] = False) -> Figure:
    """Visualize Gaze Inquiries.

    Assumes that the data is collected using BciPy and a Tobii-nano eye tracker. The default
    image used is for the matrix calibration task on a 1920x1080 screen.

    Generates a comparative matrix figure following the execution of offline analysis. Given a set of
    trailed data (left & right eye), the gaze distribution for each prompted symbol are plotted, along
    with the contour plots of mean and covariances calculated by the Gaussian Mixture Model.
    The figures may be saved or shown in a window.

    Returns a list of the figure handles created.

    Parameters
    ----------
    left_eye: (np.ndarray): Data array for the left eye data.
    right_eye: (np.ndarray): Data array for the right eye data.
    means: Optional[np.ndarray]: Means of the Gaussian Mixture Model
    covs: Optional[np.ndarray]: Covariances of the Gaussian Mixture Model
    save_path: Optional[str]: optional path to a save location of the figure generated
    show: Optional[bool]: whether or not to show the figures generated. Default: False
    img_path: Optional[str]: Image to be used as the background. Default: matrix.png
    screen_size: Optional[Tuple[int, int]]: Size of the screen used for Calibration/Copy
        Phrase tasks.
        Default: (1920, 1080)
    heatmap: Optional[bool]: Whether or not to plot the heatmap. Default: False
    raw_plot: Optional[bool]: Whether or not to plot the raw gaze data. Default: False
    """

    title = 'Raw Gaze Inquiries '

    if img_path is None:
        img_path = f'{DEFAULT_GAZE_IMAGE_PATH}'
    img = plt.imread(img_path)

    # transform the eye data to fit the display. remove > 1 values < 0 values and flip the y axis
    lx = np.clip(left_eye[:, 0], 0, 1)
    ly = np.clip(left_eye[:, 1], 0, 1)
    rx = np.clip(right_eye[:, 0], 0, 1)
    ry = np.clip(right_eye[:, 1], 0, 1)
    ly = 1 - ly
    ry = 1 - ry

    # Define mns as a copy of means to avoid modifying the original array
    mns = np.copy(means)

    if mns is not None:
        mns[:, 0] = np.clip(mns[:, 0], 0, 1)
        mns[:, 1] = np.clip(mns[:, 1], 0, 1)
        mns[:, 1] = 1 - mns[:, 1]

    # scale the eye data to the image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])

    if heatmap:
        # create a dataframe making a column for each x, y pair for both eyes and a column for the eye (left or right)
        df = pd.DataFrame({
            'x': np.concatenate((lx, rx)),
            'y': np.concatenate((ly, ry)),
            'eye': ['left'] * len(lx) + ['right'] * len(rx)
        })
        ax = sns.kdeplot(
            data=df,
            hue='eye',
            x='x',
            y='y',
            fill=False,
            thresh=0.05,
            levels=10,
            cmap="mako",
            colorbar=True)

    if raw_plot:
        ax.scatter(lx, ly, c='lightcoral', s=1)
        ax.scatter(rx, ry, c='bisque', s=1)

    if mns is not None:
        for i, (mean, cov) in enumerate(zip(mns, covs)):
            v, w = linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color='navy')
            ell.set_clip_box(ax)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(f'{title}Plot')

    if save_path is not None:
        plt.savefig(f"{save_path}/{title.lower().replace(' ', '_')}plot.png", dpi=fig.dpi)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_pupil_size(
        means: Optional[np.ndarray] = None,
        covs: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: Optional[bool] = False,
        img_path: Optional[str] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        heatmap: Optional[bool] = False,
        raw_plot: Optional[bool] = False) -> Figure:
    """Visualize Gaze Inquiries.

    Assumes that the data is collected using BciPy and a Tobii-nano eye tracker. The default
    image used is for the matrix calibration task on a 1920x1080 screen.

    Generates a comparative matrix figure following the execution of offline analysis. Given a set of
    trailed data (left & right eye), the gaze distribution for each prompted symbol are plotted, along
    with the contour plots of mean and covariances calculated by the Gaussian Mixture Model.
    The figures may be saved or shown in a window.

    Returns a list of the figure handles created.

    Parameters
    ----------
    means: Optional[np.ndarray]: Means of the Gaussian Mixture Model
    covs: Optional[np.ndarray]: Covariances of the Gaussian Mixture Model
    save_path: Optional[str]: optional path to a save location of the figure generated
    show: Optional[bool]: whether or not to show the figures generated. Default: False
    img_path: Optional[str]: Image to be used as the background. Default: matrix.png
    screen_size: Optional[Tuple[int, int]]: Size of the screen used for Calibration/Copy
        Phrase tasks.
        Default: (1920, 1080)
    heatmap: Optional[bool]: Whether or not to plot the heatmap. Default: False
    raw_plot: Optional[bool]: Whether or not to plot the raw gaze data. Default: False
    """

    title = 'Pupil Size '
    img = plt.imread(img_path)

    # Define mns as a copy of means to avoid modifying the original array
    mns = np.copy(means)

    if mns is not None:
        # mns[:, 0] = np.clip(mns[:, 0], 0, 1)
        # mns[:, 1] = np.clip(mns[:, 1], 0, 1)
        mns[:, 1] = 1 - mns[:, 1]

    # scale the eye data to the image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])

    if mns is not None:
        for i, (mean, cov) in enumerate(zip(mns, covs)):
            v, w = linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color='navy')
            ell.set_clip_box(ax)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(f'{title}Plot')

    if save_path is not None:
        plt.savefig(f"{save_path}/{title.lower().replace(' ', '_')}plot.png", dpi=fig.dpi)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_centralized_data(
        gaze_data: np.ndarray,
        save_path: Optional[str] = None,
        show: Optional[bool] = False,
        img_path: Optional[str] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        heatmap: Optional[bool] = False,
        raw_plot: Optional[bool] = False) -> Figure:
    """Visualize centralized data for each symbol.

    Assumes that the data is collected using BciPy and a Tobii-nano eye tracker. The default
    image used is for the matrix calibration task on a 1920x1080 screen.

    Generates a comparative matrix figure following the execution of offline analysis. Given a set of
    trailed data (left & right eye), the gaze distribution for ALL prompted symbols are plotted after
    the data has been centralized (i.e., mid coordinates of each symbol is subtracted from corresponding
    symbol data to populate all data around (0,0) on the screen).
    The figures may be saved or shown in a window.

    Returns a list of the figure handles created.

    Parameters
    ----------
    gaze_data: (np.ndarray): Data array for the centralized left eye data.
    save_path: Optional[str]: optional path to a save location of the figure generated
    show: Optional[bool]: whether or not to show the figures generated. Default: False
    img_path: Optional[str]: Image to be used as the background. Default: matrix.png
    screen_size: Optional[Tuple[int, int]]: Size of the screen used for Calibration/Copy Phrase tasks.
        Default: (1920, 1080)
    heatmap: Optional[bool]: Whether or not to plot the heatmap. Default: False
    raw_plot: Optional[bool]: Whether or not to plot the raw gaze data. Default: False
    """
    title = 'Centralized Data for All Symbols '

    if img_path is None:
        img_path = f'{DEFAULT_GAZE_IMAGE_PATH}'
    img = plt.imread(img_path)

    # scale the eye data to the image
    fig, ax = plt.subplots()

    # Transform the eye data to fit the display. Clip values > 1 and < -1
    # The idea here is to have the center at (0,0)
    dx = np.clip(gaze_data[:, 0], -1, 1)
    dy = np.clip(gaze_data[:, 1], -1, 1)
    ax.imshow(img, extent=[-1, 1, -1, 1])

    if raw_plot:
        ax.scatter(dx, dy, c='b', s=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(f'{title}Plot')

    if save_path is not None:
        plt.savefig(f"{save_path}/{title.lower().replace(' ', '_')}plot.png", dpi=fig.dpi)

    if show:
        plt.show()

    return fig


def visualize_results_all_symbols(
        left_eye_all: List[np.ndarray],
        right_eye_all: List[np.ndarray],
        means_all: List[np.ndarray],
        covs_all: List[np.ndarray],
        save_path: Optional[str] = None,
        show: Optional[bool] = False,
        img_path: Optional[str] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        heatmap: Optional[bool] = False,
        raw_plot: Optional[bool] = False) -> Figure:
    """Visualize results for all symbols.

    Assumes that the data is collected using BciPy and a Tobii-nano eye tracker. The default
    image used is for the matrix calibration task on a 1920x1080 screen.

    Generates a comparative matrix figure following the execution of offline analysis. Given a set of
    trailed data (left & right eye), the gaze distribution for all prompted symbols are shown in a
    single image, along with mean-cov contours of gaze distribution with respect to each symbol.
    The figures may be saved or shown in a window.

    Returns a list of the figure handles created.

    Parameters
    ----------
    left_eye_all: (np.ndarray): Data array for the left eye data, for all symbols.
    right_eye_all: (np.ndarray): Data array for the right eye data, for all symbols.
    means_all: Optional[np.ndarray]: Means of the Gaussian Mixture Model, for all symbols
    covs_all: Optional[np.ndarray]: Covariances of the Gaussian Mixture Model, for all symbols
    save_path: Optional[str]: optional path to a save location of the figure generated
    show: Optional[bool]: whether or not to show the figures generated. Default: False
    img_path: Optional[str]: Image to be used as the background. Default: matrix.png
    screen_size: Optional[Tuple[int, int]]: Size of the screen used for Calibration/Copy
        Phrase tasks.
        Default: (1920, 1080)
    heatmap: Optional[bool]: Whether or not to plot the heatmap. Default: False
    raw_plot: Optional[bool]: Whether or not to plot the raw gaze data. Default: False
    """

    title = 'Map of Gaze Data and Contours '

    if img_path is None:
        img_path = f'{DEFAULT_GAZE_IMAGE_PATH}'
    img = plt.imread(img_path)

    # scale the eye data to the image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])

    for _, (left_eye, right_eye, means, covs) in enumerate(zip(left_eye_all, right_eye_all, means_all, covs_all)):
        # transform the eye data to fit the display. remove > 1 values < 0 values and flip the y axis
        lx = np.clip(left_eye[:, 0], 0, 1)
        ly = np.clip(left_eye[:, 1], 0, 1)
        rx = np.clip(right_eye[:, 0], 0, 1)
        ry = np.clip(right_eye[:, 1], 0, 1)
        ly = 1 - ly
        ry = 1 - ry

        # Define mns as a copy of means to avoid modifying the original array
        mns = np.copy(means)

        if mns is not None:
            mns[:, 0] = np.clip(mns[:, 0], 0, 1)
            mns[:, 1] = np.clip(mns[:, 1], 0, 1)
            mns[:, 1] = 1 - mns[:, 1]

        if heatmap:
            # create a dataframe making a column for each x, y pair for both eyes and
            # a column for the eye (left or right)
            df = pd.DataFrame({
                'x': np.concatenate((lx, rx)),
                'y': np.concatenate((ly, ry)),
                'eye': ['left'] * len(lx) + ['right'] * len(rx)
            })
            ax = sns.kdeplot(
                data=df,
                hue='eye',
                x='x',
                y='y',
                fill=False,
                thresh=0.05,
                levels=10,
                cmap="mako",
                colorbar=True)

        if raw_plot:
            ax.scatter(lx, ly, c='lightcoral', s=1)
            ax.scatter(rx, ry, c='bisque', s=1)

        if mns is not None:
            for i, (mean, cov) in enumerate(zip(mns, covs)):
                v, w = linalg.eigh(cov)
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color='navy')
                ell.set_clip_box(ax)
                ell.set_alpha(0.5)
                ax.add_artist(ell)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(f'{title}Plot')

    if save_path is not None:
        plt.savefig(f"{save_path}/{title.lower().replace(' ', '_')}plot.png", dpi=fig.dpi)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_csv_eeg_triggers(trigger_col: Optional[int] = None):
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
    raw_data, _ = data.by_channel()

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

    logger.info('Press Ctrl + C to exit!')
    # Show us the figure! Depending on your OS / IDE this may not close when
    #  The window is closed, see the message above
    plt.show()


def visualize_joint_average(
        epochs: Tuple[Epochs],
        labels: List[str],
        plot_joint_times: Optional[List[float]] = [-0.1, 0, 0.2, 0.3, 0.35, 0.4, 0.5],
        save_path: Optional[str] = None,
        show: Optional[bool] = False) -> List[Figure]:
    """Visualize Joint Average.

    Plots channel average for each condition and provides corresponding topomaps for times defined in plot_joint_times.

    See: https://mne.tools/dev/generated/mne.Evoked.html#mne.Evoked.plot_joint

    Parameters:
        epochs(Tuple[Epochs]): MNE Epochs
        labels(List[str]): List of labels for each Epoch
        plot_joint_times(List[float]): List of times to plot topomaps
        save_path(Optional[str]): Path to save the figures
        show(Optional[bool]): Whether to show the figures

    Note: assumes the number of Epochs (passed as a tuple) is equal to the number of labels provided

    Returns:
        List of figures generated
    """
    assert len(epochs) == len(labels), "The number of epochs must match labels in Visualize Joint Average"

    figs = []
    for i, label in enumerate(labels):
        avg = epochs[i].average()
        fig = avg.plot_joint(times=plot_joint_times, title=label, show=show)
        if save_path:
            fig.savefig(
                f'{save_path}/{label.lower()}_topomap.png'
            )
        figs.append(fig)
    return figs


def visualize_evokeds(epochs: Tuple[Epochs, Epochs],
                      save_path: Optional[str] = None,
                      show: Optional[bool] = False) -> List[Figure]:
    """Visualize Evokeds.

    Using MNE Epochs, generate a compare evokeds plot using the mean and showing parametric confidence
    interval in shaded region.

    See: https://mne.tools/stable/generated/mne.viz.plot_compare_evokeds.html

    Parameters:

        epochs(Tuple[Epochs]): MNE Epochs Note: Assumes first epoch is nontarget and second is target.
        save_path(Optional[str]): Path to save the figures
        show(Optional[bool]): Whether to show the figures
    """
    evokeds = dict(nontarget=list(epochs[0].iter_evoked()),
                   target=list(epochs[1].iter_evoked()))
    fig = mne.viz.plot_compare_evokeds(evokeds, combine='mean', show=show)
    if save_path:
        fig[0].savefig(f'{save_path}/average_erp.png')

    return fig


def visualize_session_data(
        session_path: str,
        parameters: Union[dict, Parameters],
        show=True,
        save=True) -> Figure:
    """Visualize Session Data.

    This method is used to load and visualize EEG data after a session.

    Currently, it plots target / nontarget ERPs. It filters out TriggerTypes event,
    fixation, preview, and bcipy internal triggers. It uses the filter parameters
    used during the session to filter the data before plotting.

    Input:
        session_path(str): Path to the session directory
        parameters(dict): Dictionary of parameters
        show(bool): Whether to show the figure

    Output:
        Figure of Session Data
    """
    logger.info(f"Visualizing session data at {session_path}")
    # extract all relevant parameters
    trial_window = parameters.get("trial_window")

    raw_data = load_raw_data(str(Path(session_path, f'{RAW_DATA_FILENAME}.csv')))
    channels = raw_data.channels
    sample_rate = raw_data.sample_rate
    daq_type = raw_data.daq_type

    transform_params: ERPTransformParams = parameters.instantiate(ERPTransformParams)

    devices.load(Path(session_path, DEFAULT_DEVICE_SPEC_FILENAME))
    device_spec = devices.preconfigured_device(daq_type)

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=transform_params.notch_filter_frequency,
        bandpass_low=transform_params.filter_low,
        bandpass_high=transform_params.filter_high,
        bandpass_order=transform_params.filter_order,
        downsample_factor=transform_params.down_sampling_rate,
    )
    # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=device_spec.static_offset,
        trigger_path=f"{session_path}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
        device_type='EEG',
    )
    assert "nontarget" in trigger_targetness, "No nontarget triggers found."
    assert "target" in trigger_targetness, "No target triggers found."
    assert len(trigger_targetness) == len(trigger_timing), "Trigger targetness and timing must be the same length."

    labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]
    channel_map = analysis_channels(channels, device_spec)

    return visualize_erp(
        raw_data,
        channel_map,
        trigger_timing,
        labels,
        trial_window,
        transform=default_transform,
        plot_average=True,
        plot_topomaps=True,
        save_path=session_path if save else None,
        show=show,
    )


def visualize_gaze_accuracies(accuracy_dict: Dict[str, np.ndarray],
                              accuracy: float,
                              save_path: Optional[str] = None,
                              show: Optional[bool] = False) -> Figure:
    """
    Visualize Gaze Accuracies.

    Plot the accuracies of each symbol using a bar plot.

    Returns a list of the figure handles created.
    """
    title = 'Overall Accuracy: '

    fig, ax = plt.subplots()
    ax.bar(accuracy_dict.keys(), accuracy_dict.values())
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Accuracy')
    ax.set_title(title + str(round(accuracy, 2)))

    if save_path is not None:
        plt.savefig(f"{save_path}/{title.lower().replace(' ', '_').replace(':', '')}plot.png", dpi=fig.dpi)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def erp():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize ERP data')

    parser.add_argument(
        '-s', '--session_path',
        type=str,
        help='Path to the session directory',
        required=True)
    parser.add_argument(
        '-p', '--parameters',
        type=str,
        help='Path to the parameters file',
        default=DEFAULT_PARAMETERS_PATH)
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the figure',
        default=False)
    parser.add_argument(
        '--save',
        action='store_true',
        help='Whether to save the figure', default=True)

    args = parser.parse_args()

    parameters = load_json_parameters(args.parameters, value_cast=True)
    visualize_session_data(args.session_path, parameters, args.show, args.save)
