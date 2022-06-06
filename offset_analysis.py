"""GUI tool to visualize offset analysis when analyzing system latency."""
import csv
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import normaltest, describe
from textwrap import wrap
from bcipy.acquisition.device_info import DeviceInfo

ALLOWABLE_TOLERANCE = 0.01
RECOMMEND = False # set static to 0.0 to get good base recommendations
STATIC = 0.062
RSVP = True
PLOT = True


def channel_data(raw_data, device_info, channel_name, n_records=None):
    """Get data for a single channel.
    Parameters:
    -----------
        raw_data - complete list of samples
        device_info - metadata
        channel_name - channel for which to get data
        n_records - if present, limits the number of records returned.
    """
    if channel_name not in device_info.channels:
        print(f"{channel_name} column not found; no data will be returned")
        return []
    channel_index = device_info.channels.index(channel_name)
    arr = np.array(raw_data)
    if n_records:
        return arr[:n_records, channel_index]
    return arr[:, channel_index]


def clock_seconds(device_info: DeviceInfo, sample: int) -> float:
    """Convert the given raw_data sample number to acquisition clock
    seconds."""
    assert sample > 0
    return sample / device_info.fs


def calculate_latency(raw_data, device_info, triggers, title="", recommend_static=False, plot=False):
    """Plot raw_data triggers, including the TRG_device_stream data
    (channel streamed from the device; usually populated from a trigger box),
    as well as TRG data populated from the LSL Marker Stream. Also plots data
    from the triggers.txt file if available.

    Parameters:
    -----------
        raw_data - complete list of samples read in from the raw_data.csv file.
        device_info - metadata about the device including the sample rate.
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


    # Plot column used for photodiode; this is a continuous line with >trigger_value indicating light. Ex. TRG or AUX_1
    trg_box_channel = channel_data(raw_data, device_info, 'AUX_1')
    trg_box_y = []
    trg_ymax = 1.5
    trg_box_x = []

    # setup some analysis variable
    diode_enc = False
    starts = []
    lengths = []
    length = 0
    # Amount of voltage needed to trigger a diode
    trigger_value = 100000
    convert_triggers_values = True

    for i, val in enumerate(trg_box_channel):
        timestamp = clock_seconds(device_info, i + 1)
        value = int(float(val))
        trg_box_x.append(timestamp)
        if not convert_triggers_values:
            trg_box_y.append(value)

        if value > trigger_value and not diode_enc:
            diode_enc = True
            if convert_triggers_values:
                trg_box_y.append(1)
            starts.append(timestamp)

        elif value > trigger_value and diode_enc:
            length += 1
            if convert_triggers_values:
                trg_box_y.append(1)

        elif value <= trigger_value and diode_enc:
            diode_enc = False
            lengths.append(length)
            length = 0
            if convert_triggers_values:
                trg_box_y.append(0)
        else:
            if convert_triggers_values:
                trg_box_y.append(0)

    ax.plot(trg_box_x, trg_box_y, label='TRG (trigger box)')

    # Plot triggers.txt data if present; vertical line for each value.
    if triggers:
        trigger_diodes_timestamps = [stamp for (_name, _trgtype, stamp) in triggers if _name == '\u25A0']
        plt.vlines(trigger_diodes_timestamps,
                   ymin=-1.0, ymax=trg_ymax, label='triggers.txt (adjusted)',
                   linewidth=0.5, color='cyan')

    errors = []
    diffs = []
    x = 0
    # it may be necessary to remove the first start when calculating diffs. The wait screen can cause a false positive at times
    starts.pop(0)
    for trigger_stamp, diode_stamp in zip(trigger_diodes_timestamps, starts):
        diff = trigger_stamp - diode_stamp

        # rsvp bug with fixation
        if RSVP:
            if x > 0:
                if abs(diff) > ALLOWABLE_TOLERANCE:
                    errors.append(f'trigger={trigger_stamp} diode={diode_stamp} diff={diff}')
                diffs.append(diff)
            if x == 4:
                x = 0
            else:
                x+=1
        else:
            if abs(diff) > ALLOWABLE_TOLERANCE:
                errors.append(f'trigger={trigger_stamp} diode={diode_stamp} diff={diff}')
            diffs.append(diff)

    if recommend_static:
        # test for normality
        _, p_value = normaltest(diffs)
        print(f'{describe(diffs)}')

        # if it's not normal, take the median
        if p_value < 0.05:
            print('Non-normal distribution')
        recommended_static = abs(np.median(diffs))
        print(f'System recommended static offset median=[{recommended_static}]')
        recommended_static = abs(np.mean(diffs))
        print(f'System recommended static offset mean=[{recommended_static}]')

    else:
        if errors:
            num_errors = len(errors)
            print(f'RESULTS: Allowable tolerance between triggers and photodiode exceeded. {errors}'
                  f'Number of violations: {num_errors}')
        else:
            print(f'RESULTS: Triggers and photodiode timestamps within limit of [{ALLOWABLE_TOLERANCE}s]!')


    # Add labels for TRGs
    first_trg = trigger_diodes_timestamps[0]

    # Set initial zoom to +-5 seconds around the calibration_trigger
    if first_trg:
        ax.set_xlim(left=first_trg - 5, right=first_trg + 5)

    ax.grid(axis='x', linestyle='--', color="0.5", linewidth=0.4)
    plt.legend(loc='lower left', fontsize='small')
    # display the plot
    if plot:
        plt.show()

    # import pdb; pdb.set_trace()


def file_data(path):
    """Reads raw_data; returns raw_data and device_info."""
    with open(path) as csvfile:
        # read metadata
        r1 = next(csvfile)
        name = r1.strip().split(",")[1]
        r2 = next(csvfile)
        freq = float(r2.strip().split(",")[1])

        reader = csv.reader(csvfile)
        channels = next(reader)

        # read the rest of the lines into a list
        data = []
        for line in reader:
            data.append(line)

    device_info = DeviceInfo(fs=freq, channels=channels, name=name)
    return (data, device_info)


def read_triggers(triggers_file):
    """Read in the triggers.txt file. Convert the timestamps to be in
    aqcuisition clock units using the offset listed in the file (last entry).
    Returns:
    --------
        list of (symbol, targetness, stamp) tuples."""

    with open(triggers_file, encoding='utf-8') as trgfile:
        records = [line.split(' ') for line in trgfile.readlines()]
        (_cname, _ctype, cstamp) = records[0]
        records.pop(0)
        # (_acq_name, _acq_type, acq_stamp) = records[-1]
        static_offset = STATIC
        offset = float(cstamp) + static_offset

        corrected = []
        for i, (name, trg_type, stamp) in enumerate(records):
            if i < len(records) - 1:
                # omit offset record for plotting
                corrected.append((name, trg_type, float(stamp) + offset))
        return corrected

def main(path: str):
    """Run the viewer gui

    Parameters:
    -----------
        data_file - raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much the data is downsampled. A factor of 1
            displays the raw data.
    """
    data_file = os.path.join(path, 'raw_data.csv')
    trg_file = os.path.join(path, 'triggers.txt')
    data, device_info = file_data(data_file)
    triggers = read_triggers(trg_file)

    calculate_latency(
        data,
        device_info,
        triggers,
        title=pathlib.Path(path).name,
        recommend_static=RECOMMEND,
        plot=PLOT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graphs trigger data from a bcipy session to visualize system latency"
    )
    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)
    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    main(path)