"""GUI tool to visualize offset analysis when analyzing system latency.

Assumptions:
    - The raw_data.csv file is available.
    - The raw_data.csv file contains a TRG column, which is a continuous line of 0s and values greater than 0
        (indicating a photodiode trigger event).
    - The triggers.txt file is available.
    - All data are in the same directory and in the BciPy session data format.

Usage:
    - Run the script from the command line.
    - Select the directory containing the raw_data.csv and triggers.txt files.
    - The script will plot the TRG column from the raw_data.csv file, as well as the triggers.txt file.
    - The script will also print out the results of the analysis, including the number of violations
        (set using the --tolerance flag) and the recommended static offset (-r).

    Example usage to recommend a static offset:
        python offset_analysis.py -p 'C:/Users/test_data/test_offset_analysis_session/' -r

    Example usage to recommend a static offset using the RSVP Time Test Calibrations:
        python offset_analysis.py -p 'C:/Users/test_data/test_offset_analysis_session/' --rsvp -r

    Example usage to plot the data next to photodiode results with a static offset applied:
        python offset_analysis.py -p 'C:/Users/test_data/test_offset_analysis_session/' --offset 0.01

    Example usage to plot the data next to photodiode results with a static offset applied and a tolerance of 10 ms:
        python offset_analysis.py -p 'C:/Users/test_data/test_offset_analysis_session/' --offset 0.01 --tolerance 0.01
"""
from pathlib import Path
from textwrap import wrap
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import describe, normaltest

from bcipy.gui.file_dialog import ask_directory
from bcipy.helpers.raw_data import RawData, load
from bcipy.config import RAW_DATA_FILENAME, TRIGGER_FILENAME

# TODO: visualize the lsl timestamp differences in the csv file
def lsl_timestamp_diffs(raw_data: RawData, plot: bool = False):
    # using the lsl timestamps, calculate the difference between each timestamp
    # and plot the differences
    lsl_time_stamps = raw_data.dataframe['lsl_timestamp']
    diffs = np.diff(lsl_time_stamps)
    if plot:
        plt.plot(diffs)
        plt.show()

    return diffs

def lsl_timestamp_sample_diffs(raw_data: RawData, plot: bool = False):
    # using the lsl timestamps, calculate the difference between each timestamp
    # and plot the differences
    lsl_time_stamps = raw_data.dataframe['lsl_timestamp'].values
    first_sample = lsl_time_stamps[0]
    last_sample = lsl_time_stamps[-1]
    lsl_sample_diff = last_sample - first_sample
    # get the count of all the samples
    sample_time = raw_data.dataframe.shape[0] / raw_data.sample_rate
    print(f'LSL Timestamp Sample Count: {lsl_sample_diff} EEG Sample Count: {sample_time}')
    return lsl_sample_diff, sample_time

def clock_seconds(sample_rate: float, sample: int) -> float:
    """Convert the given raw_data sample number to acquisition clock
    seconds."""
    assert sample > 0
    return sample / sample_rate


def calculate_latency(raw_data: RawData,
                      triggers: List[tuple],
                      title: str = "",
                      recommend_static: bool = False,
                      plot: bool = False,
                      tolerance: float = 0.01,
                      rsvp: bool = False):
    """Plot raw_data triggers, including the TRG_device_stream data
    (channel streamed from the device; usually populated from a trigger box),
    as well as TRG data populated from the LSL Marker Stream. Also plots data
    from the triggers.txt file if available.

    Parameters:
    -----------
        raw_data - complete list of samples read in from the raw_data.csv file.
        triggers - list of (trg, trg_type, stamp) values from the triggers.txt
            file. Stamps have been converted to the acquisition clock using
            the offset recorded in the triggers.txt.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('acquisition clock (secs)')
    plt.ylabel('TRG value')
    if title:
        plt.title("\n".join(wrap(title, 50)))

    # Plot TRG column; this is a continuous line with >0 indicating light.
    trg_box_channel = raw_data.dataframe['TRG']
    trg_box_y = []
    trg_ymax = 1.5
    trg_box_x = []

    # setup some analysis variable
    diode_enc = False
    starts = []
    lengths = []
    length = 0

    for i, val in enumerate(trg_box_channel):
        timestamp = clock_seconds(raw_data.sample_rate, i + 1)
        value = int(float(val))
        trg_box_x.append(timestamp)
        trg_box_y.append(value)

        if value > 0 and not diode_enc:
            diode_enc = True
            starts.append(timestamp)

        if value > 0 and diode_enc:
            length += 0

        if value < 1 and diode_enc:
            diode_enc = False
            lengths.append(length)
            length = 0

    ax.plot(trg_box_x, trg_box_y, label='TRG (trigger box)')

    # Plot triggers.txt data if present; vertical line for each value.
    if triggers:
        trigger_diodes_timestamps = [
            stamp for (_name, _trgtype, stamp) in triggers if _name == '\u25A0'
        ]
        plt.vlines(trigger_diodes_timestamps,
                   ymin=-1.0,
                   ymax=trg_ymax,
                   label=f'{TRIGGER_FILENAME} (adjusted)',
                   linewidth=0.5,
                   color='cyan')

    errors = []
    diffs = []
    x = 0

    # In the case of ending the session before the triggers are finished or the photo diode is detected at the end, 
    # we balance the timestamps
    if (len(starts) != len(trigger_diodes_timestamps)):
        if len(starts) > len(trigger_diodes_timestamps):
            starts = starts[:len(trigger_diodes_timestamps)]
        else:
            trigger_diodes_timestamps = trigger_diodes_timestamps[:len(starts)]

    for trigger_stamp, diode_stamp in zip(trigger_diodes_timestamps, starts):
        diff = trigger_stamp - diode_stamp

        # RSVP and Matrix Calibration Task: Correct for fixation causing a false positive
        if x > 0:
            if abs(diff) > tolerance:
                errors.append(
                    f'trigger={trigger_stamp} diode={diode_stamp} diff={diff}'
                )
            diffs.append(diff)
        if x == 4:
            x = 0
        else:
            x += 1

    if recommend_static:
        # test for normality
        _, p_value = normaltest(diffs)
        print(f'{describe(diffs)}')

        # if it's not normal, take the median
        if p_value < 0.05:
            print('Non-normal distribution')
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
         # Add labels for TRGs
        first_trg = trigger_diodes_timestamps[0]

        # Set initial zoom to +-5 seconds around the calibration_trigger
        if first_trg:
            ax.set_xlim(left=first_trg - 5, right=first_trg + 5)

        ax.grid(axis='x', linestyle='--', color="0.5", linewidth=0.4)
        plt.legend(loc='lower left', fontsize='small')
        plt.show()

        # plot a scatter of the differences
        plt.scatter(range(len(diffs)), diffs)
        plt.show()


def read_triggers(triggers_file: str, static_offset: float):
    """Read in the triggers.txt file. Convert the timestamps to be in
    acquisition clock units using the offset listed in the file (last entry).
    Returns:
    --------
        list of (symbol, targetness, stamp) tuples."""

    
    with open(triggers_file, encoding='utf-8') as trgfile:
        records = [line.split(' ') for line in trgfile.readlines()]
        (_cname, _ctype, cstamp) = records[0]
        records.pop(0)
        # (_acq_name, _acq_type, acq_stamp) = records[-1]
        offset = float(cstamp) + static_offset

        corrected = []
        for i, (name, trg_type, stamp) in enumerate(records):
            if i < len(records) - 1:
                # omit offset record for plotting
                corrected.append((name, trg_type, float(stamp) + offset))
        return corrected


def main(data_dir: str, recommend: bool = False, static_offset: float = 0.105,
         tolerance: float = 0.01, rsvp: bool = False):
    """Run the viewer gui

    Parameters:
    -----------
        data_dir - path to the data directory, containing raw_data.csv and triggers.txt
        recommend - whether to recommend a static offset. If True, plots are not displayed.
        static_offset - fixed offset applied to triggers for plotting and analysis.
    """
    if recommend:
        static_offset = 0.0
    raw_data = load(Path(data_dir, f'{RAW_DATA_FILENAME}.csv'))
    triggers = read_triggers(Path(data_dir, f'{TRIGGER_FILENAME}'), static_offset)

    # breakpoint()
    lsl_timestamp_diffs(raw_data, plot=True)
    lsl_timestamp_sample_diffs(raw_data, plot=False)

    calculate_latency(raw_data,
                      triggers,
                      title=Path(data_dir).name,
                      recommend_static=recommend,
                      plot=not recommend,
                      tolerance=tolerance,
                      rsvp=rsvp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graphs trigger data from a bcipy session to visualize system latency")
    parser.add_argument('-p',
                        '--path',
                        help='path to the data directory',
                        default=None)
    parser.add_argument('-r',
                        '--recommend',
                        help='flag used for recommending a static offset. If set, plots are not displayed',
                        action='store_true')
    parser.add_argument('--offset',
                        help='static offset applied to triggers for plotting and analysis',
                        default=0.088)
    parser.add_argument('-t', '--tolerance',
                        help='Allowable tolerance between triggers and photodiode. Deafult 10 ms',
                        default=0.015)


    args = parser.parse_args()
    path = args.path
    if not path:
        path = ask_directory()

    main(path, bool(args.recommend), float(args.offset), float(args.tolerance))
