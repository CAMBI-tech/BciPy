"""Tools for analyzing system latency after running a timing_verification
task."""

import sqlite3


def raw_data_trg(rawdata_db: str):
    """Extract the TRG column from the raw_data buffer where there are
    non-zero values.

    Parameters:
    -----------
        rawdata_db - path to sqlite database
    Returns:
    --------
        list of (float, str), where float is the sample number
    """
    cursor = sqlite3.connect(rawdata_db, check_same_thread=False).cursor()
    cursor.execute('select timestamp, TRG from data where TRG <> 0;')
    return cursor.fetchall()


def normalized_raw_data_trg(raw_data_db: str, sample_rate_hz: float = 300.0):
    """Extract the TRG values from the raw_data buffer and and convert
    sample number to seconds.

    Parameters:
    -----------
        rawdata_db - path to sqlite database
    
    Returns:
    --------
        list of (float, str), where float is the seconds elapsed since
            start of acquisition.
    """
    data = raw_data_trg(raw_data_db)
    return [((trg[0] / sample_rate_hz), trg[1]) for trg in data]


def raw_data_sensor(raw_data_db: str, sample_rate_hz: float):
    """Extract TRG_device_stream column from the raw_data buffer if present.

    Parameters:
    ----------
        rawdata_db - path to sqlite database
    """
    pass


def read_triggers(triggers_file: str, include_trg_type: bool = False):
    """Read in the triggers.txt file. Convert the timestamps to be in
    aqcuisition clock units using the offset listed in the file (last entry).

    Parameters:
    -----------
        triggers_file - triggers.txt path
        include_trg_type - if True adds the trg type (calib, first-pres-target, 
            target, non-target, etc.)
    Returns:
    --------
        list of (float, str), where float is the seconds elapsed since
            start of acquisition.
        
        or 

        list of (float, str, targetness)
        """

    with open(triggers_file) as trgfile:
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