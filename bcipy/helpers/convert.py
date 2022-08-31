"""Functionality for converting the bcipy raw data output to other formats"""

import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from pyedflib import FILETYPE_EDFPLUS, EdfWriter
from tqdm import tqdm

from bcipy.config import RAW_DATA_FILENAME, TRIGGER_FILENAME, DEFAULT_PARAMETER_FILENAME
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.raw_data import RawData
from bcipy.helpers.system_utils import RAW_DATA_FILENAME, TRIGGER_FILENAME, DEFAULT_PARAMETER_FILENAME
from bcipy.signal.process import Composition
from bcipy.helpers.triggers import trigger_decoder, trigger_durations

import mne
from mne.io import RawArray


logger = logging.getLogger(__name__)

FILE_LENGTH_LIMIT = 150


def convert_to_edf(data_dir: str,
                   edf_path: str = None,
                   overwrite=False,
                   write_targetness=False,
                   use_event_durations=False) -> Path:
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
    use_event_durations - optional; if True assigns a duration to each event.

    Returns
    -------
        Path to new edf file
    """
    if not edf_path:
        edf_path = Path(data_dir, f'{RAW_DATA_FILENAME}.edf')

    params = load_json_parameters(Path(data_dir, DEFAULT_PARAMETER_FILENAME),
                                  value_cast=True)
    data = load_raw_data(Path(data_dir, f'{RAW_DATA_FILENAME}.csv'))
    raw_data, _ = data.by_channel()
    durations = trigger_durations(params) if use_event_durations else {}

    trigger_type, trigger_timing, trigger_label = trigger_decoder(
        str(Path(data_dir, TRIGGER_FILENAME)), remove_pre_fixation=False)

    # validate annotation parameters given data length and trigger count
    validate_annotations(len(raw_data[0]) / data.sample_rate, len(trigger_type))

    triggers = compile_triggers(
        trigger_label, trigger_type, trigger_timing, write_targetness)

    events = edf_annotations(triggers, durations)

    return write_edf(edf_path, raw_data, data.channels, data.sample_rate, events, overwrite)


def validate_annotations(record_time: float, trigger_count: int) -> None:
    """Validate Annotations.

    Using the pyedflib library, it is recommended the number of triggers (or annotations) not exceed the recording
        time in seconds. This may not result in an unsuccessful export.
    """
    if trigger_count > record_time:
        logger.warning(
            f'\n*Warning* The number of triggers [{trigger_count}] exceeds recording time [{record_time}]. '
            'Not all triggers may be written. '
            'Validate export carefully.')


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
    sample_rate - sample frequency
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
    digital_min, digital_max = [-32768, 32767]
    physical_min, physical_max = [raw_data.min(), raw_data.max()]

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
                'sample_frequency': sample_rate,
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
        channel_map: List[int],
        transform: Optional[Composition] = None,
        montage: str = 'standard_1020') -> RawArray:
    """Convert to MNE.

    Returns BciPy RawData as an MNE RawArray. This assumes all data channels are eeg and channel names
    are reflective of standard 1020 locations.
    See https://mne.tools/dev/generated/mne.channels.make_standard_montage.html
    """
    data, channels, fs = raw_data.by_channel_map(channel_map, transform)
    channel_types = ['eeg' for _ in channels]

    info = mne.create_info(channels, fs, channel_types)
    mne_data = RawArray(data, info)
    ten_twenty_montage = mne.channels.make_standard_montage(montage)
    mne_data.set_montage(ten_twenty_montage)

    return mne_data
