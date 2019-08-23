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
    cursor.execute(f'select timestamp, {column} from data where {column} <> 0;')
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

def all_sensor_data(data_dir, sample_rate_hz, column: str="TRG_device_stream"):
    """Optical sensor data connected to trigger box normalized to seconds
    since the start of acquisition."""
    return normalized_data(data_dir, sample_rate_hz, column)

def sensor_data(data_dir, sample_rate_hz, column: str="TRG_device_stream"):
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
        compressed.append((sample/sample_rate_hz, value))
        last_sample = sample

    return compressed

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


class LatencyData():
    def __init__(self, data_dir: str):
        self.sample_rate = read_sample_rate(data_dir)
        self.triggers = triggers(data_dir)
        self.lsl_trg = lsl_trg_data(data_dir, self.sample_rate)
        self.sensor = sensor_data(data_dir, self.sample_rate)

    def combined(self):
        """Combined values for triggers.txt and raw_data.csv TRG column"""
        output = []
        for i in range(len(self.triggers)):
            trg_time, stim = self.triggers[i]
            lsl_time, _stim = self.lsl_trg[i]

            # TODO: assert stims are equal?
            output.append((stim, trg_time, lsl_time))
        frame = pd.DataFrame.from_records(
            data=output, columns=["stimulus", "triggers.txt", "raw_data_TRG"])

        # Since raw_data is written after the trigger is pushed to the LSL
        # marker stream, we assume that there is some latency between the push
        # and the write operation. However, due to converting into acquisition
        # seconds, this is not always the case.
        frame['LSL_diff'] = frame['raw_data_TRG'] - frame['triggers.txt']

        # use .describe() on the resulting dataframe to see statistics.
        return frame
