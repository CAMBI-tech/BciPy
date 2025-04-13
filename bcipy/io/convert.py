# mypy: disable-error-code="no-redef"
"""Functionality for converting the bcipy raw data output to other formats"""
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple
import glob

import mne
from enum import Enum
from mne.io import RawArray
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description
from tqdm import tqdm

from bcipy.acquisition.devices import preconfigured_device
from bcipy.config import (RAW_DATA_FILENAME,
                          TRIGGER_FILENAME, SESSION_LOG_FILENAME)
from bcipy.io.load import load_raw_data
from bcipy.core.raw_data import RawData, get_1020_channel_map
from bcipy.core.triggers import trigger_decoder, TriggerType
from bcipy.signal.process import Composition

logger = logging.getLogger(SESSION_LOG_FILENAME)

FILE_LENGTH_LIMIT = 150


class ConvertFormat(Enum):

    BV = 'BrainVision'
    EDF = 'EDF'
    FIF = 'FIF'
    EEGLAB = 'EEGLAB'

    def __str__(self):
        return self.value

    @staticmethod
    def all():
        return [format for format in ConvertFormat]

    @staticmethod
    def values():
        return [format.value for format in ConvertFormat]


def convert_to_bids(
        data_dir: str,
        participant_id: str,
        session_id: str,
        run_id: str,
        output_dir: str,
        task_name: Optional[str] = None,
        line_frequency: float = 60,
        format: ConvertFormat = ConvertFormat.BV,
        label_duration: float = 0.5,
        full_labels: bool = True) -> Path:
    """Convert to BIDS.

    Convert the raw data to the Brain Imaging Data Structure (BIDS) format.
    The BIDS format is a standard for organizing and describing neuroimaging data.
    See: https://bids.neuroimaging.io/.

    Currently, this function only supports EEG data.

    Parameters
    ----------
    data_dir - path to the directory containing the raw data, triggers, and parameters
    participant_id - the participant ID
    session_id - the session ID
    run_id - the run ID
    output_dir - the directory to save the BIDS formatted data
    task_name - the name of the task
    line_frequency - the line frequency of the data (50 or 60 Hz)
    format - the format to convert the data to (BrainVision, EDF, FIF, or EEGLAB)
    label_duration - the duration of the trigger labels in seconds. Default is 0.5 seconds.
    full_labels - if True, include the full trigger labels in the BIDS data. Default is True. If False, only include
        the targetness labels (target/non-target).

    Returns
    -------
    The path to the BIDS formatted data
    """
    # validate the inputs before proceeding
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory={data_dir} does not exist")
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError as e:
            raise OSError(f"Failed to create output directory={output_dir}") from e
    if format not in ConvertFormat.all():
        raise ValueError(f"Unsupported format={format}")
    if line_frequency not in [50, 60]:
        raise ValueError("Line frequency must be 50 or 60 Hz")

    # create file paths for raw data, triggers, and parameters
    raw_data_file = os.path.join(data_dir, f'{RAW_DATA_FILENAME}.csv')
    trigger_file = os.path.join(data_dir, TRIGGER_FILENAME)

    # load the raw data and specifications for the device used to collect the data
    raw_data = load_raw_data(raw_data_file)
    channel_map = get_1020_channel_map(raw_data.channels)
    device_spec = preconfigured_device(raw_data.daq_type)
    volts = True if device_spec.channel_specs[0].units == 'volts' else False

    # load the triggers without removing any triggers other than system triggers (default)
    trigger_targetness, trigger_timing, trigger_labels = trigger_decoder(
        trigger_path=trigger_file,
        offset=device_spec.static_offset,
        device_type='EEG',
        exclusion=[TriggerType.PREVIEW]
    )

    # convert the raw data to MNE format
    mne_data = convert_to_mne(raw_data, volts=volts, channel_map=channel_map)

    # add the trigger annotations to the MNE data
    targetness_annotations = mne.Annotations(
        onset=trigger_timing,
        duration=[label_duration] * len(trigger_timing),
        description=trigger_targetness,
    )

    if full_labels:
        label_annotations = mne.Annotations(
            onset=trigger_timing,
            duration=[label_duration] * len(trigger_timing),
            description=trigger_labels,
        )
        mne_data.set_annotations(targetness_annotations + label_annotations)
    else:
        mne_data.set_annotations(targetness_annotations)
    # add the line frequency to the MNE data
    mne_data.info["line_freq"] = line_frequency

    # create the BIDS path for the data
    bids_path = BIDSPath(
        subject=participant_id,
        session=session_id,
        task=task_name,
        run=run_id,
        datatype="eeg",
        root=output_dir
    )

    # use the BIDS conversion function from MNE-BIDS
    write_raw_bids(
        mne_data,
        bids_path,
        format=format.value,
        allow_preload=True,
        overwrite=True)

    return bids_path.directory


def convert_eyetracking_to_bids(
        raw_data_path,
        output_dir,
        participant_id,
        session_id,
        run_id,
        task_name) -> str:
    """Converts the raw eye tracking data to BIDS format.

    There is currently no standard for eye tracking data in BIDS. This function will write the raw eye tracking data
    to a tsv file in a BIDS-style format.

    Parameters
    ----------
    raw_data_path : str
        Path to the raw eye tracking data
    output_dir : str
        Path to the output directory.
        This should be where other BIDS formatted data is stored for the participant, session, and run.
    participant_id : str
        Participant ID, e.g. 'S01'
    session_id : str
        Session ID, e.g. '01'
    run_id : str
        Run ID, e.g. '01'
    task_name : str
        Task name. Example: 'RSVPCalibration'

    Returns
    -------
    str
        Path to the BIDS formatted eye tracking data
    """
    # check that the raw data path exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw eye tracking data path={raw_data_path} does not exist")

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory={output_dir} does not exist")

    found_files = glob.glob(f"{raw_data_path}/eyetracker*.csv")
    if len(found_files) == 0:
        raise FileNotFoundError(f"No raw eye tracking data found in directory={raw_data_path}")
    if len(found_files) > 1:
        raise ValueError(f"Multiple raw eye tracking data files found in directory={raw_data_path}")

    eye_tracking_file = found_files[0]
    logger.info(f"Found raw eye tracking data file={eye_tracking_file}")

    # load the raw eye tracking data
    raw_data = load_raw_data(eye_tracking_file)
    # get the data as a pandas DataFrame
    data = raw_data.dataframe

    # make the et subdirectory
    et_dir = os.path.join(output_dir, 'et')
    os.makedirs(et_dir, exist_ok=True)

    # write the dataframe as a tsv file to the output directory
    output_filename = f'sub-{participant_id}_ses-{session_id}_task-{task_name}_run-{run_id}_eyetracking.tsv'
    output_path = os.path.join(et_dir, output_filename)
    data.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Eye tracking data saved to {output_path}")
    return output_path


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
        channel_map: Optional[List[int]] = None,
        channel_types: Optional[List[str]] = None,
        transform: Optional[Composition] = None,
        montage: str = 'standard_1020',
        volts: bool = False,
        remove_system_channels: bool = True) -> RawArray:
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
        remove_system_channels - if True, exclude the system and trigger channels from the MNE RawArray
            (last two channels in BciPy data).

    Returns
    -------
        MNE RawArray
    """
    # if no channel map provided, assume all channels are included
    if not channel_map:
        # if remove_system_channels is True, exclude the system and trigger channels (last two channels)
        if remove_system_channels:
            channel_map = [1] * (len(raw_data.channels) - 2)
            channel_map.extend([0, 0])  # exclude the system and trigger channels
        else:
            channel_map = [1] * len(raw_data.channels)

    data, channels, fs = raw_data.by_channel_map(channel_map, transform)

    # if no channel types provided, assume all channels are eeg
    if not channel_types:
        logger.warning("No channel types provided. Assuming all channels are EEG.")
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
