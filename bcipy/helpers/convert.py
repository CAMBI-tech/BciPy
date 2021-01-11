"""Functionality for converting the bcipy raw data output to other formats"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyedflib import FILETYPE_EDFPLUS, EdfWriter

from bcipy.helpers.load import load_json_parameters, read_data_csv
from bcipy.helpers.triggers import read_triggers, trigger_durations


def convert_to_edf(data_dir: str,
                   edf_path: str = None,
                   overwrite=False,
                   use_event_durations=False) -> Path:
    """ Converts BciPy raw_data to the EDF+ filetype using pyEDFlib.

    Parameters
    ----------
    data_dir - directory which contains the data to be converted. This
        location must also contain a parameters.json configuration file.
    edf_path - optional path to write converted data; defaults to writing
        a file named raw.edf in the data_dir.
    overwrite - If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    use_event_durations - optional; if True assigns a duration to each event.

    Returns
    -------
        Path to new edf file
    """
    if not edf_path:
        edf_path = Path(data_dir, 'raw.edf')

    params = load_json_parameters(Path(data_dir, 'parameters.json'),
                                  value_cast=True)
    raw_data, _, ch_names, _, sfreq = read_data_csv(
        Path(data_dir, params['raw_data_name']))
    durations = trigger_durations(params) if use_event_durations else {}

    with open(Path(data_dir, params.get('trigger_file_name', 'triggers.txt')), 'r') as trg_file:
        triggers = read_triggers(trg_file)
    events = edf_annotations(triggers, durations)

    return write_edf(edf_path, raw_data, ch_names, sfreq, events, overwrite)


def write_edf(output_path: str,
              raw_data: np.array,
              ch_names: List[str],
              sfreq: float,
              events: List[Tuple[float, float, str]],
              overwrite=False) -> Path:
    """
    Converts BciPy raw_data to the EDF+ filetype using pyEDFlib.

    Adapted from: https://github.com/holgern/pyedflib

    Parameters
    ----------
    output_path - optional path to write converted data; defaults to writing
        a file named raw.edf in the raw_data_dir.
    raw_data - raw data with a row for each channel
    ch_names - names of the channels
    sfreq - sample frequency
    events - List[Tuple(onset_in_seconds: float, duration_in_seconds: float, description: str)]
    overwrite - If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.

    Returns
    -------
        Path to new edf file
    """
    if not overwrite and os.path.exists(output_path):
        raise OSError('EDF file already exists.')

    # set conversion parameters
    dmin, dmax = [-32768, 32767]
    pmin, pmax = [raw_data.min(), raw_data.max()]
    n_channels = len(raw_data)

    try:
        writer = EdfWriter(str(output_path),
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

        writer.setSignalHeaders(channel_info)
        writer.writeSamples(data_list)

        if events:
            for onset, duration, label in events:
                writer.writeAnnotation(onset, duration, label)
    except Exception as error:
        logging.getLogger(__name__).info(error)
        return None
    finally:
        writer.close()
    return output_path


def edf_annotations(triggers: List[Tuple[str, str, float]],
                    durations: Dict[str, float] = {}
                    ) -> List[Tuple[float, float, str]]:
    """Convert bcipy triggers to the format expected by pyedflib for writing annotations.

    Parameters
    ----------
        triggers - trigger data in the format (symbol, targetness, stamp),
          where stamp has been converted to acquisition clock units.
        durations - optional map defining the duration (seconds) of each
            trigger type. The default is to assign 0.0 seconds.
    Returns
    -------
        List[Tuple(onset_in_seconds, duration_in_seconds, description)]
    """
    return [(timestamp, durations.get(targetness, 0.0), label)
            for (label, targetness, timestamp) in triggers]
