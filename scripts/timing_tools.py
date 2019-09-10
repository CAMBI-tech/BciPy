"""Tools for analyzing system latency after running a timing_verification
task."""

import sqlite3
import pandas as pd
from collections import namedtuple


def extract_column(data_dir, column):
    """Extract the timestamp and column from the raw_data buffer where 
    there are non-zero values.

    Parameters:
    -----------
    data_dir - path to sqlite database
    
    Returns:
    --------
    list of (float, str), where float is the sample number
    """
    db = f"{data_dir}/raw_data.db"
    conn = sqlite3.connect(db, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        f'select timestamp, {column} from data where {column} <> 0;')
    return cursor.fetchall()


def normalized_data(data_dir: str, column: str, sample_rate_hz: float):
    """Extract the timestamp and column from the raw_data buffer and and
    convert sample number to seconds elapsed since acquisition start.

    Parameters:
    -----------
        rawdata_db - path to sqlite database
        column - column of data to extract
        sample_rate_hz - frequency of data collection in hz.
    
    Returns:
    --------
        list of (float, str), where float is the seconds elapsed since
            start of acquisition.
    """
    data = extract_column(data_dir, column=column)
    return [((row[0] / sample_rate_hz), row[1]) for row in data]


def lsl_trg_data(data_dir: str, sample_rate_hz: float):
    """TRG data (LSL Marker stream written to in display module) normalized
    to acquisition seconds."""
    return normalized_data(data_dir, 'TRG', sample_rate_hz)


def all_sensor_data(data_dir,
                    sample_rate_hz,
                    column: str = "TRG_device_stream"):
    """Optical sensor data connected to trigger box normalized to seconds
    since the start of acquisition."""
    return normalized_data(data_dir, column, sample_rate_hz)


def sensor_data(data_dir, sample_rate_hz, column: str = "TRG_device_stream"):
    """Optical sensor data connected to trigger box normalized to seconds
    since the start of acquisition. Compresses samples with consecutive
    triggers."""

    data = extract_column(data_dir, column)

    # The sensor channel will emit a value ('1') for as long as the stimulus
    # is displayed. For measuring latency, we are only concerned with the
    # time at which the stimulus initially appeared.

    # Collapse consecutive samples, only taking the first. Assumes that there
    # is a gap between stimuli.
    compressed = []
    last_sample = None
    for row in data:
        sample, value = row
        if last_sample and int(last_sample) == int(sample) - 1:
            # skip this one
            last_sample = sample
            continue
        # append normalized value
        compressed.append((sample / sample_rate_hz, value))
        last_sample = sample

    return compressed

def skipped_iter(data, skip_criteria):
    """Utility function that returns an iterator where data items are skipped
    according to some provided criteria.

    Parameters:
    -----------
        data - list of data items
        skip_criteria - lambda or function that takes an item and determines
            whether or not to skip.
    Returns:
    --------
        a data item iter starting at the first unskipped item.
    """
    iterator = iter(data)

    # Skip past any sensor data that was recorded prior to min time.
    # if min_time:
    item = next(iterator)
    i = 0
    while skip_criteria(item):
        item = next(iterator)
        i = i + 1
    return iter(data[i:])

def filled_sensor_data(sensor_data, stimuli, min_time=None, includes_calibration=False):
    """
    Correlates the sensor data to the stimulus.

    Parameters:
    ----------
        sensor_data - list of sensor data where each value is a 
            (timestamp, value) tuple which represents the starting
            time (in acquisition clock) at which the sensor was activated.
        stimuli - complete list of stimulus presented.
        min_time - if provided, discards any sensor data recorded before this time. 
            Should be in given in the normalized acquisition clock time.
    Returns:
    --------
        expanded list of tuples (timestamp, stim) values, with a timestamp
            for each stim. Assumes that the sensor was in place during the
            entire experiment.
    """
    detectable_items = ['x', '■']
    undetectable = ['○', '□']

    if includes_calibration:
        detectable_items.append('calibration_trigger')
    else:
        undetectable.append('calibration_trigger')

    expanded = []
    sensor_iter = iter(sensor_data)

    # Skip past any sensor data that was recorded prior to min time.
    if min_time:
        sensor_iter = skipped_iter(
            sensor_data,
            skip_criteria=lambda sensor_item: sensor_item[0] < min_time)

    for stim in stimuli:
        if stim in undetectable:
            expanded.append((None, stim))
        elif stim in detectable_items:
            sensor_time = next(sensor_iter)[0]
            expanded.append((sensor_time, stim))
        else:
            raise f"Unknown stimulus: {stim}"

    try:
        next(sensor_iter)
    except StopIteration:
        return expanded

    print("Not all sensor records were consumed.")
    return expanded


def triggers(data_dir: str, include_trg_type: bool = False):
    """Read in the triggers.txt file. Convert the timestamps to be in
    aqcuisition clock units using the offset listed in the file (last entry).

    Parameters:
    -----------
        data_dir - data directory with the triggers.txt file
        include_trg_type - if True adds the trg type (calib, first-pres-target, 
            target, non-target, etc.)
    Returns:
    --------
        list of (float, str), where float is the seconds elapsed since
            start of acquisition.
        
        or 

        list of (float, str, targetness)
        """

    with open(f"{data_dir}/triggers.txt") as trgfile:
        records = [line.split(' ') for line in trgfile.readlines()]
        (_cname, _ctype, cstamp) = records[0]
        (_acq_name, _acq_type, acq_stamp) = records[-1]
        offset = float(acq_stamp) - float(cstamp)

        corrected = []
        for i, (name, trg_type, stamp) in enumerate(records):
            if i < len(records) - 1:
                # omit offset record for plotting
                seconds = (float(stamp) + offset)
                record = (seconds, name)
                if include_trg_type:
                    record = (seconds, name, trg_type)
                corrected.append(record)
        return corrected


def read_sample_rate(data_dir: str):
    """Read the sample rate from the raw_data.csv file"""
    with open(f"{data_dir}/raw_data.csv") as csvfile:
        _name_row = next(csvfile)
        fs = float(next(csvfile).strip().split(",")[1])
        return fs

def calib_time(normalized_triggers):
    """Given a list of normalized triggers, return the time (acquisition) clock
    when the calibration trigger was displayed."""
    for acq_time, stim in normalized_triggers:        
        if stim == 'calibration_trigger':
            return acq_time
    return None


class LatencyData():
    def __init__(self, data_dir: str):
        self.sample_rate = read_sample_rate(data_dir)
        self.triggers = triggers(data_dir)
        self.lsl_trg = lsl_trg_data(data_dir, self.sample_rate)

        self.all_sensor_data = all_sensor_data(data_dir, self.sample_rate)
        self.sensor_data = sensor_data(data_dir, self.sample_rate)

        stim = [trg[1] for trg in self.triggers]
        min_time = calib_time(self.triggers)
        self.filled_sensors = filled_sensor_data(self.sensor_data, stim, min_time=min_time)

    def combined(self):
        """Combined values for triggers.txt and raw_data.csv TRG column"""
        output = []
        for i in range(len(self.triggers)):
            trg_time, stim = self.triggers[i]
            lsl_time, _stim = self.lsl_trg[i]
            sensor_time, _stim = self.filled_sensors[i]

            # TODO: assert stims are equal?
            output.append((stim, trg_time, lsl_time, sensor_time))
        frame = pd.DataFrame.from_records(
            data=output,
            columns=["stimulus", "triggers.txt", "raw_data_TRG", "sensor"])

        frame['diff'] = abs(frame['raw_data_TRG'] - frame['triggers.txt'])

        frame['triggers_offset'] = frame['sensor'] - frame['triggers.txt']
        frame['rawdata_offset'] = frame['sensor'] - frame['raw_data_TRG']

        # use .describe() on the resulting dataframe to see statistics.
        return frame


def main(path: str, outpath:str):
    """Run the viewer gui

    Parameters:
    -----------
        data_file - raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much the data is downsampled. A factor of 1
            displays the raw data.
    """
    data = LatencyData(path)
    frame = data.combined()
    if outpath:
        frame.to_csv(path_or_buf=outpath, index=False)
        print(f"Data written to: {outpath}")

    print(frame.describe())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Generates latency data from a bcipy session"
    )
    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)
    parser.add_argument('-o', '--out', help='path to output file; if not provided, only summary data is output', default=None)
    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import filedialog
        from tkinter import *
        root = Tk()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    main(path, args.out)
