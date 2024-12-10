from typing import Any, List, Tuple
from textwrap import wrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import normaltest

from bcipy.io.load import load_raw_data, ask_directory, load_json_parameters
from bcipy.core.raw_data import RawData
from bcipy.core.triggers import trigger_decoder, TriggerType

from bcipy.config import (
    TRIGGER_FILENAME,
    DIODE_TRIGGER,
    RAW_DATA_FILENAME,
    DEFAULT_TRIGGER_CHANNEL_NAME,
    DEFAULT_PARAMETERS_FILENAME
)


def sample_to_seconds(sample_rate: float, sample: int) -> float:
    """Convert the given raw_data sample number to seconds."""
    assert sample > 0, f"Sample # must be greater than 0. Got {sample}"
    assert sample_rate > 0, f"Sample rate must be greater than 0. Got {sample_rate}"
    return sample / sample_rate


def calculate_latency(raw_data: RawData,
                      diode_channel: str,
                      triggers: List[Tuple[Any, Any, Any]],
                      stim_length: int,
                      plot_title: str = "",
                      recommend_static: bool = False,
                      plot: bool = False,
                      tolerance: float = 0.01,
                      correct_diode_false_positive: int = 0) -> Tuple[List[float], List[str]]:
    """Calculate the latency between the photodiode and the trigger timestamps.

    This method uses collected photodiode information and trigger timestamps from
    BciPy Time Test Tasks to determine system offsets that need to be accounted for before usage.

    This can be determined using the recommend_static parameter, which will provide a recommended
    static offset based on the median and mean of the differences between the photodiode and trigger timestamps.

    Using a tolerance value, the method reports any differences between the photodiode and trigger
    that would be considered excessive for data collection.

    Note: Often photodiode information is collected from a trigger box, and
    provided as a channel in the raw_data. This has been tested using
    Wearable Sensing's and Brain Vision's trigger boxes with the photodiode
    connected. The photodiode channel is expected to be a continuous signal with
    values >0 indicating light.

    Plots:
    ------
    - Plot triggers and photodiode data to visualize the differences between the
    two. The photodiode data is plotted as a continuous line with values >0
    indicating light. The triggers are plotted as vertical lines at the
    timestamps provided in the triggers.txt file.

    - LSL timestamp differences are plotted as a scatter plot to visualize any delays from
    the acquisition of data. All values should be within the sample rate of the data.

    - Offset differences are plotted as a scatter plot to visualize the differences between the
    photodiode and trigger timestamps over the course of the session. This can be used to determine
    if there are any patterns of delay or offset that need to be accounted for.

    Parameters:
    -----------
        raw_data - complete list of samples read in from the raw_data.csv file.
        diode_channel - the channel name for the photodiode in the raw_data.
        triggers - list of (trg, trg_type, stamp) values from the triggers.txt
            file. Stamps have been converted to the acquisition clock using
            the offset recorded in the triggers.txt.
        plot_title - title for the plot.
        recommend_static - if True, recommend a static offset using the median and mean.
        plot - if True, display the plot.
        tolerance - the allowable difference between the photodiode and trigger timestamps for reporting.

    Returns:
    --------
        diffs - list of differences between the photodiode and trigger timestamps.
        errors - list of differences that exceed the tolerance.
    """
    # Photodiode column; this is a continuous line with >0 indicating light.
    trg_box_channel = raw_data.dataframe[diode_channel]

    # Initialize variables for photodiode data extraction
    trg_box_x = []
    trg_box_y = []
    trg_ymax = 1.5
    diode_enc = False
    starts = []

    # Get the photodiode trigger data in the form of a list of timestamps
    for i, val in enumerate(trg_box_channel):
        timestamp = sample_to_seconds(raw_data.sample_rate, i + 1)
        value = int(float(val))
        trg_box_x.append(timestamp)
        trg_box_y.append(value)

        if value > 0 and not diode_enc:
            diode_enc = True
            starts.append(timestamp)

        if value > 0 and diode_enc:
            pass

        if value < 1 and diode_enc:
            diode_enc = False

    # Plot triggers.txt data if present; vertical line for each value.
    if triggers:
        trigger_diodes_timestamps = [
            stamp for (_name, _trgtype, stamp) in triggers if _name == DIODE_TRIGGER
        ]

    # If the photodiode is falsing detected at the beginning of the session, we remove the first timestamp
    if correct_diode_false_positive > 0:
        starts = starts[correct_diode_false_positive:]

    # In the case of ending the session before the triggers are finished or the photo diode is detected at the end,
    # we balance the timestamps
    if (len(starts) != len(trigger_diodes_timestamps)):
        if len(starts) > len(trigger_diodes_timestamps):
            starts = starts[:len(trigger_diodes_timestamps)]
        else:
            trigger_diodes_timestamps = trigger_diodes_timestamps[:len(starts)]

    # Check for differences between the photodiode and trigger timestamps.
    # Store any differences that exceed the tolerance.
    errors = []
    diffs = []
    x = 0

    for trigger_stamp, diode_stamp in zip(trigger_diodes_timestamps, starts):
        diff = trigger_stamp - diode_stamp

        # TODO: RSVP and Matrix Calibration Task: Correct for prompt/fixation causing a false positive
        if x > 0:
            if abs(diff) > tolerance and not recommend_static:
                errors.append(
                    f'trigger={trigger_stamp} diode={diode_stamp} diff={diff}'
                )
            diffs.append(diff)

        # Reset x to 0 if we are at the end of the inquiry.
        if x == (stim_length / 2) - 1:
            x = 0

        else:
            x += 1

    if recommend_static:
        # test for normality
        _, p_value = normaltest(diffs)

        # if it's not normal, take the median
        if p_value < 0.05:
            print(f'Non-normal distribution of diffs. p-value=[{p_value}] Consider using median for static offset.')
        recommended_static = abs(np.median(diffs))
        print(
            f'System recommended static offset median=[{recommended_static}]')
        recommended_static = abs(np.mean(diffs))
        print(f'System recommended static offset mean=[{recommended_static}]')

    else:
        if errors:
            num_errors = len(errors)
            print(
                f'RESULTS: Allowable tolerance between triggers and photodiode exceeded. {errors}'
                f'Number of violations: {num_errors}')
        else:
            print(
                f'RESULTS: Triggers and photodiode timestamps within limit of [{tolerance}s]!'
            )

    # display the plot
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.title("\n".join(wrap(plot_title, 50)))
        plt.xlabel('acquisition clock (secs)')
        plt.ylabel(f'{diode_channel} (photodiode)')

        if triggers:
            plt.vlines(trigger_diodes_timestamps,
                       ymin=-1.0,
                       ymax=trg_ymax,
                       label=f'{TRIGGER_FILENAME} (adjusted)',
                       linewidth=0.5,
                       color='cyan')

        ax.plot(trg_box_x, trg_box_y, label=f'{diode_channel} (photodiode triggers)')

        # Add labels for TRGs
        first_trg = trigger_diodes_timestamps[0]

        # Set initial zoom to +-5 seconds around the calibration_trigger
        if first_trg:
            ax.set_xlim(left=first_trg - 5, right=first_trg + 5)

        ax.grid(axis='x', linestyle='--', color="0.5", linewidth=0.4)
        plt.legend(loc='lower left', fontsize='small')
        plt.show()

        lsl_timestamp_diffs(raw_data, plot=plot)
        sample_rate_diffs(raw_data)

        # plot a scatter of the differences
        plt.scatter(range(len(diffs)), diffs)
        plt.show()

    return diffs, errors


def lsl_timestamp_diffs(raw_data: RawData, plot: bool = False) -> List[float]:
    """Calculate the differences between the LSL timestamps and plot the differences.

    Parameters:
    -----------
        raw_data - BciPy RawData object.
        plot - if True, display a plot of the differences between LSL timestamps.

    Returns:
    --------
        diffs - list of differences between the LSL timestamps.
    """
    # using the lsl timestamps, calculate the difference between each timestamp
    lsl_time_stamps = raw_data.dataframe['lsl_timestamp']
    diffs = np.diff(lsl_time_stamps)

    # display the plot if requested. This will show the differences between each timestamp,
    # which should be consistent.
    if plot:
        plt.plot(diffs)
        plt.show()

    return diffs.tolist()


def sample_rate_diffs(raw_data: RawData) -> Tuple[int, float]:
    """Calculate the expected time recorded from the raw_data and LSL timestamps.

    Differences in this time may indicate a problem with sample loss.

    Parameters:
    -----------
        raw_data - BciPy RawData object.

    Returns:
    --------
        lsl_sample_diff - the difference between the first and last LSL timestamps (seconds).
        sample_time - all samples recorded from raw_data (seconds).
    """
    # using the lsl timestamps and samples, calculate the expected time recorded.

    # extract the first and last lsl timestamps to get the total time recorded
    lsl_time_stamps = raw_data.dataframe['lsl_timestamp'].values
    first_sample = lsl_time_stamps[0]
    last_sample = lsl_time_stamps[-1]
    lsl_sample_diff = last_sample - first_sample

    # get the count of all the samples and calculate the time recorded from the raw_data
    sample_time = raw_data.dataframe.shape[0] / raw_data.sample_rate
    print(f'LSL Timestamp Sample Count: {lsl_sample_diff} EEG Sample Count: {sample_time}')
    return lsl_sample_diff, sample_time


def extract_data_latency_calculation(
        data_dir: str,
        recommend: bool,
        static_offset: float) -> Tuple[RawData, List[Tuple[Any, Any, Any]], float]:
    """Extract the raw data and triggers from the data directory.

    If recommend is True, a static offset is recommended. Any static offset provided as
    an arg will be set to zero to calculate the recommended offset.

    Parameters:
    -----------
        data_dir - path to the data directory, containing raw_data.csv and triggers.txt
        recommend - whether to recommend a static offset. If True, plots are not displayed.
        static_offset - fixed offset applied to triggers for plotting and analysis.
    """
    # We set the static offset to zero if we are recommending a new offet value
    if recommend:
        static_offset = 0.0

    raw_data = load_raw_data(str(Path(data_dir, f'{RAW_DATA_FILENAME}.csv')))
    trigger_targetness, trigger_timing, trigger_label = trigger_decoder(
        trigger_path=str(Path(data_dir, f'{TRIGGER_FILENAME}')),
        exclusion=[TriggerType.FIXATION],
        remove_pre_fixation=False,
        offset=static_offset,
        device_type='EEG'
    )
    triggers = list(zip(trigger_label, trigger_targetness, trigger_timing))
    return raw_data, triggers, static_offset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Graphs trigger data from a bcipy session to visualize system latency.'
                    'Recommends a static offset if requested.')
    parser.add_argument('-d',
                        '--data_path',
                        help='Path to the data directory',
                        default=None)
    parser.add_argument('-p',
                        '--plot',
                        help='Flag used for plotting the data. If set, plots are displayed',
                        default=False,
                        action='store_true')
    parser.add_argument('-r',
                        '--recommend',
                        help='Flag used for recommending a static offset. If set, plots are not displayed',
                        default=False,
                        action='store_true')
    parser.add_argument('-trg',
                        help='Argument to change the photodiode trigger name. '
                        f'Default is {DEFAULT_TRIGGER_CHANNEL_NAME}',
                        default=DEFAULT_TRIGGER_CHANNEL_NAME,
                        type=str,
                        dest='trigger_name')
    parser.add_argument('--offset',
                        help='The static offset applied to triggers for plotting and analysis',
                        default=0.09)
    parser.add_argument('-t', '--tolerance',
                        help='Allowable tolerance between triggers and photodiode. Deafult 10 ms',
                        default=0.015)
    parser.add_argument('-f', '--false_positive',
                        help='Allows to correct the false positive of the photodiode at the beginning of the session.',
                        default=0,
                        type=int)

    args = parser.parse_args()
    data_path = args.data_path
    if not data_path:
        data_path = ask_directory(prompt="Please select a BciPy time test directory..", strict=True)

    # grab the stim length from the data directory parameters
    stim_length = load_json_parameters(f'{data_path}/{DEFAULT_PARAMETERS_FILENAME}', value_cast=True)['stim_length']

    raw_data, triggers, static_offset = extract_data_latency_calculation(
        data_path,
        bool(args.recommend),
        float(args.offset))
    response = calculate_latency(
        raw_data,
        args.trigger_name,
        triggers,
        stim_length,
        recommend_static=args.recommend,
        plot=args.plot,
        tolerance=float(args.tolerance),
        correct_diode_false_positive=args.false_positive)
