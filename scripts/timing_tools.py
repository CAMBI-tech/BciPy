"""Tools for analyzing system latency after running a timing_verification
task."""

import sqlite3


def raw_data_trg(rawdata_db):
    """Extract the TRG column from the raw_data buffer where there are
    non-zero values"""
    cursor = sqlite3.connect(rawdata_db, check_same_thread=False).cursor()
    cursor.execute('select timestamp, TRG from data where TRG <> 0;')
    return cursor.fetchall()

def normalized_raw_data_trg(raw_data_db, sample_rate_hz=300):
    data = raw_data_trg(raw_data_db)
    # TODO: divide values by sample_rate_hz to get seconds
    return data

def raw_data_sensor(raw_data_db):
    """Extract TRG_device_stream column from the raw_data buffer if present.
    """
    pass

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