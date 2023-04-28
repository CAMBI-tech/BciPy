"""Functionality for converting the bcipy raw data output to other formats"""
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from pyedflib import FILETYPE_EDFPLUS, EdfWriter, FILETYPE_BDFPLUS
from tqdm import tqdm

from bcipy.config import RAW_DATA_FILENAME, TRIGGER_FILENAME, DEFAULT_PARAMETER_FILENAME
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.raw_data import RawData
from bcipy.signal.process import Composition, get_default_transform
from bcipy.helpers.triggers import trigger_decoder, trigger_durations

import mne
from mne.io import RawArray


logger = logging.getLogger(__name__)

FILE_LENGTH_LIMIT = 150


def convert_to_edf(data_dir: str,
                   edf_path: str = None,
                   overwrite: bool = False,
                   write_targetness: bool = False,
                   use_event_durations: bool = False,
                   remove_pre_fixation: bool = False,
                   pre_filter: bool = False) -> Path:
    """ Converts BciPy data to the EDF+ filetype using pyEDFlib.

    See https://www.edfplus.info/ for more detailed information about the edf specification.
    See https://www.teuniz.net/edfbrowser/ for a free EDF/BDF viewer.

    Parameters
    ----------
    data_dir - directory which contains the data to be converted. This location must contain:
        1. a parameter file.
        2. a raw_data.csv file.
        3. a trigger.txt file.
    edf_path - optional; path to write converted data; defaults to writing a file named raw_data.edf in the data_dir.
        Must end in .edf.
    overwrite - If True, the destination file (if it exists) will be overwritten. If False (default), an error will
        be raised if the file exists.
    write_targetness - If True, and targetness information is available, write that instead of the stimuli markers.
        False by default.
    use_event_durations - optional; if True assigns a duration to each event.
    remove_pre_fixation - optional; if True removes the non-inquiry based markers from the trigger data.
    pre_filter - optional; if True, apply the default filter to the data using to loaded parameters file.

    Returns
    -------
        Path to new edf file
    """
    if not edf_path:
        edf_path = str(Path(data_dir, f'{RAW_DATA_FILENAME}.edf').resolve())
    if not edf_path.endswith('.edf'):
        raise ValueError(f'edf_path=[{edf_path}] must end in .edf')

    data, channels, fs, events, annotation_channels, pre_filter = pyedf_convert(
        data_dir,
        write_targetness=write_targetness,
        use_event_durations=use_event_durations,
        remove_pre_fixation=remove_pre_fixation,
        pre_filter=pre_filter)

    return write_pyedf(
        edf_path,
        data,
        channels,
        fs,
        events,
        FILETYPE_EDFPLUS,
        overwrite,
        annotation_channels,
        pre_filter)


def convert_to_bdf(data_dir: str,
                   bdf_path: str = None,
                   overwrite: bool = False,
                   write_targetness: bool = False,
                   use_event_durations: bool = False,
                   remove_pre_fixation: bool = False,
                   pre_filter: bool = False) -> Path:
    """ Converts BciPy data to the BDF+ filetype using pyEDFlib.

    See https://www.biosemi.com/faq/file_format.htm for more detailed information about the BDF specification.
    See https://www.teuniz.net/edfbrowser/ for a free EDF/BDF viewer.

    Parameters
    ----------
    data_dir - directory which contains the data to be converted. This location must contain:
        1. a parameter file.
        2. a raw_data.csv file.
        3. a trigger.txt file.
    bdf_path - optional; path to write converted data; defaults to writing a file named raw_data.edf in the data_dir.
        Must end in .edf.
    overwrite - If True, the destination file (if it exists) will be overwritten. If False (default), an error will
        be raised if the file exists.
    write_targetness - If True, and targetness information is available, write that instead of the stimuli markers.
        False by default.
    use_event_durations - optional; if True assigns a duration to each event.
    remove_pre_fixation - optional; if True removes the non-inquiry based markers from the trigger data.
    pre_filter - optional; if True, apply the default filter to the data using to loaded parameters file.

    Returns
    -------
        Path to new bdf file
    """
    if not bdf_path:
        bdf_path = str(Path(data_dir, f'{RAW_DATA_FILENAME}.bdf').resolve())
    if not bdf_path.endswith('.bdf'):
        raise ValueError(f'bdf_path=[{bdf_path}] must end in .bdf')

    data, channels, fs, events, annotation_channels, pre_filter = pyedf_convert(
        data_dir,
        write_targetness=write_targetness,
        use_event_durations=use_event_durations,
        remove_pre_fixation=remove_pre_fixation,
        pre_filter=pre_filter)

    return write_pyedf(
        bdf_path,
        data,
        channels,
        fs,
        events,
        FILETYPE_BDFPLUS,
        overwrite,
        annotation_channels,
        pre_filter)


def pyedf_convert(data_dir: str,
                  write_targetness: bool = False,
                  use_event_durations: bool = False,
                  remove_pre_fixation: bool = False,
                  pre_filter: bool = False) -> Tuple[RawData, List[str], int, List[Tuple[str, int, int]], int]:
    """ Converts BciPy data to formats that can be used by pyEDFlib.

    Parameters
    ----------

    data_dir - directory which contains the data to be converted. This location must contain:
        1. a parameter file.
        2. a raw_data.csv file.
        3. a trigger.txt file.
    write_targetness - If True, and targetness information is available, write that instead of the stimuli markers.
        False by default.
    use_event_durations - optional; if True assigns a duration to each event.
    remove_pre_fixation - optional; if True removes the non-inquiry based markers from the trigger data.
    pre_filter - optional; if True, apply the default filter to the data using to loaded parameters file.

    Returns
    -------
        data - raw data
        channels - list of channel names
        fs - sampling rate
        events - list of events
        annotation_channels - number of annotation channels
        pre_filter - False if no filter was applied,
            otherwise a string of the filter parameters used to filter the data
    """

    params = load_json_parameters(Path(data_dir, DEFAULT_PARAMETER_FILENAME),
                                  value_cast=True)
    data = load_raw_data(Path(data_dir, f'{RAW_DATA_FILENAME}.csv'))
    fs = data.sample_rate
    if pre_filter:
        default_transform = get_default_transform(
            sample_rate_hz=data.sample_rate,
            notch_freq_hz=params.get("notch_filter_frequency"),
            bandpass_low=params.get("filter_low"),
            bandpass_high=params.get("filter_high"),
            bandpass_order=params.get("filter_order"),
            downsample_factor=params.get("down_sampling_rate"),
        )
        raw_data, fs = data.by_channel(transform=default_transform)
        pre_filter = (f"HP:{params.get('filter_low')} LP:{params.get('filter_high')}"
                      f"N:{params.get('notch_filter_frequency')} D:{params.get('down_sampling_rate')} "
                      f"O:{params.get('filter_order')}")
    else:
        raw_data, _ = data.by_channel()
    durations = trigger_durations(params) if use_event_durations else {}
    static_offset = params['static_trigger_offset']
    logger.info(f'Static offset: {static_offset}')

    trigger_type, trigger_timing, trigger_label = trigger_decoder(
        str(Path(data_dir, TRIGGER_FILENAME)),
        remove_pre_fixation=remove_pre_fixation,
        offset=static_offset)

    # validate annotation parameters given data length and trigger count, up the annotation limit byt increasing
    # annotation_channels if needed
    annotation_channels = validate_annotations(len(raw_data[0]) / data.sample_rate, len(trigger_type))

    triggers = compile_triggers(
        trigger_label, trigger_type, trigger_timing, write_targetness)

    events = compile_annotations(triggers, durations)

    return raw_data, data.channels, fs, events, annotation_channels, pre_filter


def validate_annotations(record_time: float, trigger_count: int) -> int:
    """Validate Annotations.

    Using the pyedflib library, it is recommended the number of triggers (or annotations) not exceed the recording
        time in seconds. This may result in an unsuccessful export. The best workaround is to increase the number of
        annotation channels. This function will return the number of annotation channels needed to export the triggers
        successfully. The maximum number of annotation channels is 64 and it is not advised to use more than required
        due to fize size restrictions.

    See https://github.com/holgern/pyedflib/issues/34 for more information.
    """
    if trigger_count > record_time:
        logger.warning(
            f'\n*Warning* The number of triggers [{trigger_count}] exceeds recording time [{record_time}]. ')
        annotation_channels = round(trigger_count / record_time) + 1
        logger.warning(f'\nIncreasing annotation channels to {annotation_channels} to compensate.')
        return annotation_channels
    return 1


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


def write_pyedf(output_path: str,
                raw_data: np.array,
                ch_names: List[str],
                sample_rate: float,
                events: List[Tuple[float, float, str]],
                file_type: str = FILETYPE_EDFPLUS,
                overwrite: bool = False,
                annotation_channels: int = 1,
                pre_filter: Optional[str] = None) -> Path:
    """
    Converts BciPy raw_data to the EDF+ or BDF+ filetype using pyEDFlib.

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
    annotation_channels - number of annotation channels to use
    pre_filter - optional; if provided, add string of filters applied to channel data

    Returns
    -------
        Path to new edf or bdf file
    """
    if not overwrite and os.path.exists(output_path):
        raise OSError(f'{output_path} already exists.')

    physical_min, physical_max = [raw_data.min(), raw_data.max()]
    n_channels = len(raw_data)

    try:
        writer = EdfWriter(output_path, n_channels=n_channels, file_type=file_type)

        channel_info = []
        data_list = []

        for i in range(n_channels):
            ch_dict = {
                'label': ch_names[i],
                'dimension': 'uV',
                'sample_frequency': sample_rate,
                'physical_min': physical_min,
                'physical_max': physical_max,
            }

            if pre_filter:
                ch_dict['prefilter'] = pre_filter

            channel_info.append(ch_dict)
            data_list.append(raw_data[i])

        if events:
            writer.set_number_of_annotation_signals(annotation_channels)
            for onset, duration, label in events:
                writer.writeAnnotation(onset, duration, label)

        writer.setSignalHeaders(channel_info)
        writer.writeSamples(data_list)

    except Exception as error:
        logger.info(error)
        raise error
    finally:
        writer.close()
    return output_path


def compile_annotations(triggers: List[Tuple[str, str, float]],
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


def compress(tar_file_name: str, members: List[str]) -> None:
    """
    File compression and archiving.

    Adds files to a tar archive and compresses them using the gzip compression
        format.

    Parameters
    ----------
    tar_file_name (str): name of resulting compressed tar archive. Input string
        can either include or not include file extension (.tar.gz), as this will
        be checked.
    members (List[str]): list of files and folders to be compressed into archive.
        Each file or folder requires the relative file path and full file extension.
        Individual file paths and names that are very long may throw an error
        upon extraction, proceed with caution.

    Returns
    -------
        None
    """

    for member in members:
        # Checks if file exists
        if not os.path.exists(member):
            raise FileNotFoundError(f"This file or folder, '{member}', "
                                    "does not exist!\nPlease rerun program")
        # Checks for file names that may be too long. OS level restriction.
        if len(member) > FILE_LENGTH_LIMIT:
            logger.warning(
                f'File length exceeds compression limit=[{FILE_LENGTH_LIMIT}]. '
                'This may cause issues later with extraction. Please proceed at your own discretion.')

    full_tar_name = tar_name_checker(tar_file_name)
    # Warns user that reopening tar archive that already exists will overwrite it
    if os.path.exists(full_tar_name):
        raise Exception(f"This tar archive=[{full_tar_name}] already exists, continuing will "
                        "overwrite anything in the existing archive.")

    # Opens file for gzip (gz) compressed writing (w)
    # Uses default compression level 9 (highest compression, slowest speed)
    with tarfile.open(full_tar_name, mode="w:gz") as tar:
        # Sets progress bar through tqdm
        progress = tqdm(members)
        for member in progress:
            # Adds file/folder to the tar file and compresses it
            tar.add(member)
            # Sets progress description of progress bar
            progress.set_description(f"Compressing {member}")


def decompress(tar_file: str, path: str) -> None:
    """
    Archive decompression and extraction.

    Takes .tar.gz archive, decompresses, and extracts its contents. Should only be
    used with files created by compress() method.

    Parameters
    ----------
    tar_file (str): name of .tar.gz archive to be decompressed. Input string
        can either include or not include file extension (.tar.gz), as this will
        be checked.
    path (str): file path and name of folder for the extracted contents of the
        archive. If only a name is entered, by default it will place the folder
        in the current working directory. (ex. "extracted" creates a new folder
        in the current directory and extracts the archive contents to it)

    Returns
    -------
        None
    """

    full_tar_name = tar_name_checker(tar_file)

    # Checks if file exists
    if not os.path.exists(full_tar_name):
        raise FileNotFoundError(f"This file or folder, '{tar_file}', "
                                "does not exist!\nPlease rerun program")

    # Opens file for gzip (gz) compressed reading (r)
    with tarfile.open(full_tar_name, mode="r:gz") as tar:
        members = tar.getmembers()

        # Sets progress bar through tqdm
        progress = tqdm(members)
        for member in progress:
            tar.extract(member, path=path)
            # Sets progress description of progress bar
            progress.set_description(f"Extracting {member.name}")


def archive_list(tar_file: str) -> List[str]:
    """
    Returns contents of tar archive.

    Takes .tar.gz archive and returns a full list of its contents, including
        folders, subfolders, and files, with their full relative paths and names.

    Parameters
    ----------
    tar_file (str): name of desired .tar.gz archive. Input string can either
        include or not include file extension (.tar.gz), as this will be checked.

    Returns
    -------
        List[str] of all items in archive
    """

    full_tar_name = tar_name_checker(tar_file)

    # Checks if file exists
    if not os.path.exists(full_tar_name):
        raise FileNotFoundError(f"This file or folder, '{tar_file}', "
                                "does not exist!\nPlease rerun program")

    # Adds names of archive contents to list
    with tarfile.open(full_tar_name, mode="r") as tar:
        tar_list = tar.getnames()
    return tar_list


def tar_name_checker(tar_file_name: str) -> str:
    """
    Checks and modifies file name for tar archive.

    Helper method that takes a tar file name and checks it for the appropriate
        file extension (".tar.gz" for now), returning it if it already does, and
        adding the extension and then returning it if it does not.

    Parameters
    ----------
    tar_file_name (str): name for archive being checked.

    Returns
    -------
        String of properly formatted tar archive name
    """

    if tar_file_name.endswith('.tar.gz'):
        return tar_file_name
    return f'{tar_file_name}.tar.gz'


def convert_to_mne(
        raw_data: RawData,
        channel_map: List[int] = None,
        channel_types: Optional[List[str]] = None,
        transform: Optional[Composition] = None,
        montage: str = 'standard_1020',
        volts: bool = True) -> RawArray:
    """Convert to MNE.

    Returns BciPy RawData as an MNE RawArray. This assumes all channel names
    are reflective of provided montage locations.

    Parameters
    ----------
        raw_data - BciPy RawData object
        channel_map - optional list of channels to include in the MNE RawArray [1, 0, 1, 0, 1, 0, 1, 0].
            1 indicates the channel should be included, 0 indicates it should be excluded.
            Must be the same length as the number of channels in the raw_data.
        channel_types - list of channel types to include in the MNE RawArray.
            If None, all channels will be assumed to be eeg.
            See: https://mne.tools/stable/overview/implementation.html#supported-channel-types
        transform - optional transform to apply to the data
        montage - name of the channel location montage to use.
            See https://mne.tools/dev/generated/mne.channels.make_standard_montage.html
        volts - if True, assume data is already in volts. If false, assume data is in microvolts and convert to volts.
            MNE expects data to be in volts.
            See: https://mne.tools/dev/overview/implementation.html#internal-representation-units
    """
    # if no channel map provided, assume all channels are included
    if not channel_map:
        channel_map = [1] * len(raw_data.channels)

    data, channels, fs = raw_data.by_channel_map(channel_map, transform)

    # if no channel types provided, assume all channels are eeg
    if not channel_types:
        channel_types = ['eeg'] * len(channels)

    # check that number of channel types matches number of channels in the case custom channel types are provided
    assert len(channel_types) == len(channels), \
        f'Number of channel types ({len(channel_types)}) must match number of channels ({len(channels)})'

    info = mne.create_info(channels, fs, channel_types)
    mne_data = RawArray(data, info)
    ten_twenty_montage = mne.channels.make_standard_montage(montage)
    mne_data.set_montage(ten_twenty_montage)

    # convert to volts if necessary (the default for many systems is microvolts)
    if not volts:
        mne_data = mne_data.apply_function(lambda x: x * 1e-6)

    return mne_data


def tobii_to_norm(tobii_units: Tuple[float, float]) -> Tuple[float, float]:
    """Tobii to PsychoPy's 'norm' units.

    https://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    https://www.psychopy.org/general/units.html

    Tobii uses an Active Display Coordinate System.
        The point (0, 0) denotes the upper left corner and (1, 1) the lower right corner of it.

    PsychoPy uses several coordinate systems, the normalized window unit is assumed here.
        The point (0, 0) denotes the center of the screen and (-1, -1) the upper left corner
        and (1, 1) the lower right corner.

    """
    # check that Tobii units are within the expected range
    assert 0 <= tobii_units[0] <= 1, "Tobii x coordinate must be between 0 and 1"
    assert 0 <= tobii_units[1] <= 1, "Tobii y coordinate must be between 0 and 1"

    # convert Tobii units to Psychopy units
    norm_x = (tobii_units[0] - 0.5) * 2
    norm_y = (tobii_units[1] - 0.5) * 2 * -1
    return (norm_x, norm_y)


def norm_to_tobii(norm_units: Tuple[float, float]) -> Tuple[float, float]:
    """PsychoPy's 'norm' units to Tobii.

    https://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    https://www.psychopy.org/general/units.html

    Tobii uses an Active Display Coordinate System.
        The point (0, 0) denotes the upper left corner and (1, 1) the lower right corner of it.

    PsychoPy uses several coordinate systems, the normalized window unit is assumed here.
        The point (0, 0) denotes the center of the screen and (-1, -1) the upper left corner
        and (1, 1) the lower right corner.
    """
    # check that the coordinates are within the bounds of the screen
    assert norm_units[0] >= -1 and norm_units[0] <= 1, "X coordinate must be between -1 and 1"
    assert norm_units[1] >= -1 and norm_units[1] <= 1, "Y coordinate must be between -1 and 1"

    # convert PsychoPy norm units to Tobii units
    tobii_x = (norm_units[0] / 2) + 0.5
    tobii_y = ((norm_units[1] * -1) / 2) + 0.5
    return (tobii_x, tobii_y)
