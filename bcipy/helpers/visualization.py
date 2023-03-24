import logging

from typing import List, Optional, Tuple
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from bcipy.config import TRIGGER_FILENAME, RAW_DATA_FILENAME
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.triggers import TriggerType, trigger_decoder
from bcipy.helpers.convert import convert_to_mne
from bcipy.helpers.load import choose_csv_file, load_raw_data
from bcipy.helpers.stimuli import mne_epochs
from bcipy.helpers.raw_data import RawData
from bcipy.signal.process import Composition, get_default_transform

import mne
from mne import Epochs
from mne.io import read_raw_edf

log = logging.getLogger(__name__)


def visualize_erp(
        raw_data: RawData,
        channel_map: List[int],
        trigger_timing: List[float],
        trigger_labels: List[int],
        trial_length: float,
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
    trial_length: length of trials described in trigger_timing / labels in seconds
    transform(Composition): Optional BciPy composition to apply to data before visualization
    plot_average: Optional[boolean]: whether or not to average over all channels. Default: True
    plot_topomap: Optional[boolean]: whether or not to plot all channels across target/nontarget. Default: True
    show: Optional[boolean]: whether or not to show the figures generated. Default: False
    save_path: optional path to a save location of the figure generated
    """
    mne_data, _ = convert_to_mne(raw_data, channel_map=channel_map, transform=transform)
    epochs = mne_epochs(mne_data, trigger_timing, trigger_labels, interval=[0, trial_length])
    # *Note* We assume, as described above, two trigger classes are defined for use in trigger_labels
    # (Nontarget=0 and Target=1). This will map into two corresponding MNE epochs whose indexing starts at 1.
    # Therefore, epochs['1'] == Nontarget and epochs['2'] == Target.
    epochs = (epochs['1'], epochs['2'])
    figs = []
    if plot_average:
        figs.extend(visualize_evokeds(epochs, save_path=save_path, show=show))
    if plot_topomaps:
        figs.extend(visualize_joint_average(epochs, ['Non-Target', 'Target'], save_path=save_path, show=show))

    return figs


def plot_edf(edf_path: str, auto_scale: Optional[bool] = False):
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


def visualize_joint_average(
        epochs: Tuple[Epochs],
        labels: List[str],
        plot_joint_times: Optional[List[float]] = [-0.1, 0, 0.2, 0.3, 0.35, 0.4, 0.5],
        save_path: Optional[str] = None,
        show: Optional[bool] = False) -> List[Figure]:
    """Visualize Joint Average.

    Plots channel average for each condition and provides corresponding topomaps for times defined in plot_joint_times.

    See: https://mne.tools/dev/generated/mne.Evoked.html#mne.Evoked.plot_joint

    Note: assumes the number of Epochs (passed as a tuple) is equal to the number of labels provided
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


def visualize_evokeds(epochs: Tuple[Epochs, Epochs], save_path: Optional[str]
                      = None, show: Optional[bool] = False) -> List[Figure]:
    """Visualize Evokeds.

    Using MNE Epochs, generate a compare evokeds plot using the mean and showing parametric confidence
    interval in shaded region.

    See: https://mne.tools/stable/generated/mne.viz.plot_compare_evokeds.html

    Note: Assumes first epoch is nontarget and second is target."""
    evokeds = dict(nontarget=list(epochs[0].iter_evoked()),
                   target=list(epochs[1].iter_evoked()))
    fig = mne.viz.plot_compare_evokeds(evokeds, combine='mean', show=show)
    if save_path:
        fig[0].savefig(f'{save_path}/average_erp.png')

    return fig


def visualize_session_data(session_path: str, parameters: dict, show=True) -> Figure:
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
    # extract all relevant parameters
    poststim_length = parameters.get("trial_length")
    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    raw_data = load_raw_data(Path(session_path, f'{RAW_DATA_FILENAME}.csv'))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    # Process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{session_path}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    assert "nontarget" in trigger_targetness, "No nontarget triggers found."
    assert "target" in trigger_targetness, "No target triggers found."
    assert len(trigger_targetness) == len(trigger_timing), "Trigger targetness and timing must be the same length."

    labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]
    channel_map = analysis_channels(channels, type_amp)

    return visualize_erp(
        raw_data,
        channel_map,
        trigger_timing,
        labels,
        poststim_length,
        transform=default_transform,
        plot_average=True,
        plot_topomaps=True,
        save_path=session_path,
        show=show,
    )
