"""Functionality for converting the bcipy raw data output to other formats"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyedflib import FILETYPE_EDFPLUS, EdfWriter

from bcipy.helpers.load import load_json_parameters, load_raw_data, extract_mode
from bcipy.helpers.triggers import trigger_decoder, apply_trigger_offset, trigger_durations


logger = logging.getLogger(__name__)


def convert_to_edf(data_dir: str,
                   edf_path: str = None,
                   overwrite=False,
                   write_targetness=False,
                   use_event_durations=False,
                   mode=False,
                   annotation_channels=None) -> Path:
    """ Converts BciPy raw_data to the EDF+ filetype using pyEDFlib.

    See https://www.edfplus.info/ for the official EDF+ spec for more detailed
        information.
    See https://www.teuniz.net/edflib_python/index.html for a free EDF viewer.

    Parameters
    ----------
    data_dir - directory which contains the data to be converted. This
        location must also contain a parameters.json configuration file.
    edf_path - optional path to write converted data; defaults to writing
        a file named raw.edf in the data_dir.
    overwrite - If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    write_targetness - If True, and targetness information is available, write
        that instead of the stimuli markers. False by default.
    mode - optional; for a given task, define the task mode. Ex. 'calibration', 'copy_phrase'.
        If not provided, it will be extracted from the data_dir.
    use_event_durations - optional; if True assigns a duration to each event.
    annotation_channels - optional; integer between 2-64 that will extend the number of
        annotations available to export. Use in cases where annotations are
        cut off.

    Returns
    -------
        Path to new edf file
    """
    if not edf_path:
        edf_path = Path(data_dir, 'raw.edf')

    params = load_json_parameters(Path(data_dir, 'parameters.json'),
                                  value_cast=True)
    data = load_raw_data(Path(data_dir, params['raw_data_name']))
    raw_data = data.by_channel()
    durations = trigger_durations(params) if use_event_durations else {}

    # If a mode override is not provided, try to extract it from the file structure
    if not mode:
        mode = extract_mode(data_dir)

    symbol_info, trial_target_info, timing_info, offset = trigger_decoder(
        mode, Path(data_dir, params.get('trigger_file_name', 'triggers.txt')), remove_pre_fixation=False)

    # validate annotation parameters given data length and trigger count
    validate_annotations(len(raw_data[0]) / data.sample_rate, len(symbol_info), annotation_channels)

    # get static and system offsets
    observed_offset = offset + params.get('static_trigger_offset', 0.0)
    trigger_timing = apply_trigger_offset(timing_info, observed_offset)

    triggers = compile_triggers(
        symbol_info, trial_target_info, trigger_timing, write_targetness)

    events = edf_annotations(triggers, durations)

    return write_edf(edf_path, raw_data, data.channels, data.sample_rate, events, overwrite, annotation_channels)


def validate_annotations(record_time: float, trigger_count: int, annotation_channels: bool) -> None:
    """Validate Annotations.

    Using the pyedflib library, it is recommended the number of triggers (or annotations) not exceed the recording
        time in seconds. This may not result in an unsuccessful export, therefore, we advise users to increase
        annotation channels incrementally as needed to avoid losing information. If the number of annotation
        channels is too high and no annotations are written to all channels created, a read error may result.
    """
    if trigger_count > record_time and not annotation_channels:
        logger.warning(
            f'\n*Warning* The number of triggers [{trigger_count}] exceeds recording time [{record_time}]. '
            'Not all triggers may be written. '
            'Validate export carefully and increase annotation_channels incrementally to add missing triggers.')


def compile_triggers(labels: List[str], targetness: List[str], timing: List[float],
                     write_targetness: bool) -> List[Tuple[str, str, float]]:
    """Compile Triggers.

    Compile trigger information in a way that we edf conversion can easily digest. (label, targetness, timing).
        If write targetness is true, use the targetness as a label.
    """
    triggers = []
    i = 0
    for label in labels:
        # if targetness information available and the flag set to true, write another trigger with target information
        if write_targetness:
            triggers.append((targetness[i], targetness[i], timing[i]))
        else:
            triggers.append((label, targetness[i], timing[i]))
        i += 1
    return triggers


def write_edf(output_path: str,
              raw_data: np.array,
              ch_names: List[str],
              sample_rate: float,
              events: List[Tuple[float, float, str]],
              overwrite=False,
              annotation_channels=None) -> Path:
    """
    Converts BciPy raw_data to the EDF+ filetype using pyEDFlib.

    Adapted from: https://github.com/holgern/pyedflib

    Parameters
    ----------
    output_path - optional path to write converted data; defaults to writing
        a file named raw.edf in the raw_data_dir.
    raw_data - raw data with a row for each channel
    ch_names - names of the channels
    sample_rate - sample frequency
    events - List[Tuple(onset_in_seconds: float, duration_in_seconds: float, description: str)]
    overwrite - If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    annotation_channels - integer between 2-64 that will extend the number of
        annotations available to export. Use in cases where annotations are
        cut off. In some viewers, as the number of these channels increase, it
        may cause other data to be trimmed. Please use with caution and examine the exports.

    Returns
    -------
        Path to new edf file
    """
    if not overwrite and os.path.exists(output_path):
        raise OSError('EDF file already exists.')

    # set conversion parameters
    digital_min, digital_max = [-32768, 32767]
    physical_min, physical_max = [raw_data.min(), raw_data.max()]

    n_channels = len(raw_data)

    try:
        writer = EdfWriter(str(output_path),
                           n_channels=n_channels,
                           file_type=FILETYPE_EDFPLUS)
        if annotation_channels:
            writer.set_number_of_annotation_signals(annotation_channels)
        channel_info = []
        data_list = []

        for i in range(n_channels):
            ch_dict = {
                'label': ch_names[i],
                'dimension': 'uV',
                'sample_rate': sample_rate,
                'physical_min': physical_min,
                'physical_max': physical_max,
                'digital_min': digital_min,
                'digital_max': digital_max,
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
