"""GUI tool to visualize offset analysis when analyzing system latency."""
import csv
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from bcipy.acquisition.device_info import DeviceInfo


def channel_data(raw_data, device_info, channel_name, n_records=None):
    """Get data for a single channel.
    Parameters:
    -----------
        raw_data - complete list of samples
        device_info - metadata
        channel_name - channel for which to get data
        n_records - if present, limits the number of records returned.
    """
    channel_index = device_info.channels.index(channel_name)
    arr = np.array(raw_data)
    if n_records:
        return arr[:n_records, channel_index]
    return arr[:, channel_index]


def plot_triggers(raw_data, device_info, triggers, title=""):
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
        plt.title(title)

    # Plot TRG_device_stream column; this is a continuous line.
    trg_box_channel = channel_data(raw_data, device_info, 'TRG_device_stream')
    trg_box_y = [int(float(val)) for val in trg_box_channel]
    # convert x-axis to seconds
    trg_box_x = [n / device_info.fs for n in range(1, len(trg_box_y) + 1)]
    ax.plot(trg_box_x, trg_box_y, label='TRG_device_stream (trigger box)')

    trg_ymax = 1.5

    # Plot triggers.txt data if present; vertical line for each value.
    if triggers:
        plt.vlines([stamp for (_name, _trgtype, stamp) in triggers],
            ymin=-1.0, ymax=trg_ymax, label='triggers.txt (adjusted)',
            linewidth=0.5, color='cyan')

    # Plot TRG column, vertical line for each one.
    trg_channel = channel_data(raw_data, device_info, 'TRG')
    trg_stamps = [(i + 1) / (device_info.fs)
                  for i, trg in enumerate(trg_channel) if trg != '0']
    plt.vlines(trg_stamps, ymin=-1.0, ymax=trg_ymax, label='TRG (LSL)',
        linewidth=0.5, color='red')

    # Add labels for TRGs
    first_trg = None
    for i, trg in enumerate(trg_channel):
        if trg != '0':
            secs = (i + 1) / (device_info.fs)
            secs_lbl = str(round(secs, 2))
            ax.annotate(s=f"{trg} ({secs_lbl}s)", xy=(secs, trg_ymax),
                fontsize='small', color='red', horizontalalignment='right',
                rotation=270)
            if not first_trg:
                first_trg = secs

    # Set initial zoom to +-5 seconds around the calibration_trigger
    if first_trg:
        ax.set_xlim(left=first_trg - 5, right=first_trg + 5)

    ax.grid(axis='x', linestyle='--', color="0.5", linewidth=0.4)
    plt.legend(loc='lower left', fontsize='small')
    # display the plot
    plt.show()


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

    with open(triggers_file) as trgfile:
        records = [line.split(' ') for line in trgfile.readlines()]
        (_cname, _ctype, cstamp) = records[0]
        (_acq_name, _acq_type, acq_stamp) = records[-1]
        offset = float(acq_stamp) - float(cstamp)

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
    
    plot_triggers(data, device_info, triggers, title=pathlib.Path(path).name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Graphs trigger data from a bcipy session to visualize system latency"
    )
    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)
    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import filedialog
        from tkinter import *
        root = Tk()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    main(path)
