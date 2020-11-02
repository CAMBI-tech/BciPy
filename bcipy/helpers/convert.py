"""Functionality for converting the bcipy raw data output to other formats"""

import logging
import os
from datetime import datetime
from typing import List, Tuple
from pathlib import Path

import numpy as np
from pyedflib import FILETYPE_EDFPLUS, EdfWriter
from bcipy.helpers.triggers import read_triggers, trigger_durations, read_triggers_from_rawdata
from bcipy.helpers.load import load_json_parameters


def bcipy_write_edf(raw_data: np.array,
                    ch_names: List[str],
                    sfreq: float,
                    fname: str,
                    events: List[Tuple[float, float, str]] = None,
                    overwrite=False):
    """
    Converts BciPy raw_data to the EDF+ filetype using pyEDFlib.

    Adapted from: https://github.com/holgern/pyedflib

    Parameters
    ----------
    raw data - np array with a row for each channel
    ch_names - names of the channels
    sfreq - sample frequency
    fname - File name of the new dataset. Filenames should end with .edf
    events : List[Tuple(onset_in_seconds: float, duration_in_seconds: float, description: str)]
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')

    date = datetime.now().strftime('%d %b %Y %H:%M:%S')

    # set conversion parameters
    dmin, dmax = [-32768, 32767]
    pmin, pmax = [raw_data.min(), raw_data.max()]
    n_channels = len(raw_data)

    try:
        writer = EdfWriter(fname,
                           n_channels=n_channels,
                           file_type=FILETYPE_EDFPLUS)
        channel_info = []
        data_list = []

        for i in range(n_channels):
            ch_dict = {
                'label': ch_names[i],
                'dimension': 'uV',
                'sample_rate': sfreq,
                'physical_min': pmin,
                'physical_max': pmax,
                'digital_min': dmin,
                'digital_max': dmax,
                'transducer': '',
                'prefilter': ''
            }

            channel_info.append(ch_dict)
            data_list.append(raw_data[i])

        writer.setTechnician('bcipy.helpers.convert')
        writer.setSignalHeaders(channel_info)
        writer.setStartdatetime(date)
        writer.writeSamples(data_list)

        if events:
            for onset, duration, label in events:
                writer.writeAnnotation(onset, duration, label)
    except Exception as error:
        logging.getLogger(__name__).info(error)
        return False
    finally:
        writer.close()
    return True


def edf_annotations(raw_data_directory: str) -> List[Tuple[float, float, str]]:
    """Convert bcipy triggers to the format expected by pyedflib for writing annotations.

    Returns
    -------
        List[Tuple(onset_in_seconds, duration_in_seconds, description)]
    """

    params = load_json_parameters(Path(raw_data_directory, 'parameters.json'),
                                  value_cast=True)
    mode = 'copy_phrase' if Path(raw_data_directory,
                                 'session.json').exists() else 'calibration'
    duration = trigger_durations(params)

    if params['acq_device'] == 'LSL' and mode == 'calibration':
        # TRG channel is more accurate when it is available.
        triggers = read_triggers_from_rawdata(raw_data_directory, params, mode)
    else:
        triggers = read_triggers(
            Path(raw_data_directory, params['trigger_file_name']))

    # Convert to format expected by EDF.
    return [(timestamp, duration[targetness], label)
            for (label, targetness, timestamp) in triggers]
