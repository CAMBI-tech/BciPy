# mypy: disable-error-code="arg-type, union-attr"
import json
import logging
import os
import pickle
from pathlib import Path
from shutil import copyfile
from time import localtime, strftime
from typing import List, Optional

from bcipy.config import (DEFAULT_ENCODING, DEFAULT_EXPERIMENT_PATH,
                          DEFAULT_FIELD_PATH, DEFAULT_PARAMETERS_PATH,
                          EXPERIMENT_FILENAME, FIELD_FILENAME, ROOT,
                          SIGNAL_MODEL_FILE_SUFFIX)
from bcipy.gui.file_dialog import ask_directory, ask_filename
from bcipy.helpers.exceptions import (BciPyCoreException,
                                      InvalidExperimentException)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.raw_data import RawData
from bcipy.preferences import preferences
from bcipy.signal.model import SignalModel

log = logging.getLogger(__name__)


def copy_parameters(path: str = DEFAULT_PARAMETERS_PATH,
                    destination: Optional[str] = None) -> str:
    """Creates a copy of the given configuration (parameters.json) to the
    given directory and returns the path.

    Parameters:
    -----------
        path: str - optional path of parameters file to copy; used default if not provided.
        destination: str - optional destination directory; default is the same
          directory as the default parameters.
    Returns:
    --------
        path to the new file.
    """
    default_dir = str(Path(DEFAULT_PARAMETERS_PATH).parent)

    destination = default_dir if destination is None else destination
    filename = strftime('parameters_%Y-%m-%d_%Hh%Mm%Ss.json', localtime())

    path = str(Path(destination, filename))
    copyfile(DEFAULT_PARAMETERS_PATH, path)
    return path


def load_experiments(path: str = f'{DEFAULT_EXPERIMENT_PATH}/{EXPERIMENT_FILENAME}') -> dict:
    """Load Experiments.

    PARAMETERS
    ----------
    :param: path: string path to the experiments file.

    Returns
    -------
        A dictionary of experiments, with the following format:
            { name: { fields : {name: '', required: bool, anonymize: bool}, summary: '' } }

    """
    with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        return json.load(json_file)


def extract_mode(bcipy_data_directory: str) -> str:
    """Extract Mode.

    This method extracts the task mode from a BciPy data save directory. This is important for
        trigger conversions and extracting targeteness.

    *note*: this is not compatible with older versions of BciPy (pre 1.5.0) where
        the tasks and modes were considered together using integers (1, 2, 3).

    PARAMETERS
    ----------
    :param: bcipy_data_directory: string path to the data directory
    """
    directory = bcipy_data_directory.lower()
    if 'calibration' in directory:
        return 'calibration'
    elif 'copy' in directory:
        return 'copy_phrase'
    raise BciPyCoreException(f'No valid mode could be extracted from [{directory}]')


def load_fields(path: str = f'{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}') -> dict:
    """Load Fields.

    PARAMETERS
    ----------
    :param: path: string path to the fields file.

    Returns
    -------
        A dictionary of fields, with the following format:
            {
                "field_name": {
                    "help_text": "",
                    "type": ""
            }

    """
    with open(path, 'r', encoding=DEFAULT_ENCODING) as json_file:
        return json.load(json_file)


def load_experiment_fields(experiment: dict) -> list:
    """Load Experiment Fields.

    {
        'fields': [{}, {}],
        'summary': ''
    }

    Using the experiment dictionary, loop over the field keys and put them in a list.
    """
    if isinstance(experiment, dict):
        try:
            return [name for field in experiment['fields'] for name in field.keys()]
        except KeyError:
            raise InvalidExperimentException(
                'Experiment is not formatted correctly. It should be passed as a dictionary with the fields and'
                f' summary keys. Fields is a list of dictionaries. Summary is a string. \n experiment=[{experiment}]')
    raise TypeError('Unsupported experiment type. It should be passed as a dictionary with the fields and summary keys')


def load_json_parameters(path: str, value_cast: bool = False) -> Parameters:
    """Load JSON Parameters.

    Given a path to a json of parameters, convert to a dictionary and optionally
        cast the type.

    Expects the following format:
    "fake_data": {
        "value": "true",
        "section": "bci_config",
        "readableName": "Fake Data Sessions",
        "helpTip": "If true, fake data server used",
        "recommended_values": "",
        "type": "bool"
        }

    PARAMETERS
    ----------
    :param: path: string path to the parameters file.
    :param: value_case: True/False cast values to specified type.

    Returns
    -------
        a Parameters object that behaves like a dict.
    """
    return Parameters(source=path, cast_values=value_cast)


def load_experimental_data() -> str:
    filename = ask_directory()  # show dialog box and return the path
    log.info("Loaded Experimental Data From: %s" % filename)
    return filename


def load_signal_models(directory: Optional[str] = None) -> List[SignalModel]:
    """Load all signal models in a given directory.

    Models are assumed to have been written using bcipy.helpers.save.save_model
    function and should be serialized as pickled files. Note that reading
    pickled files is a potential security concern so only load from trusted
    directories.

    Args:
        dirname (str, optional): Location of pretrained models. If not
            provided the user will be prompted for a location.
    """
    if not directory or Path(directory).is_file():
        directory = ask_directory()

    # update preferences
    path = Path(directory)
    preferences.signal_model_directory = str(path)

    models = []
    for file_path in path.glob(f"*{SIGNAL_MODEL_FILE_SUFFIX}"):
        with open(file_path, "rb") as signal_file:
            model = pickle.load(signal_file)
            log.info(f"Loading model {model}")
            models.append(model)
    return models


def choose_signal_models(device_types: List[str]) -> List[SignalModel]:
    """Prompt the user to load a signal model for each provided device.

    Parameters
    ----------
        device_types - list of device content types (ex. 'EEG')
    """
    return [
        model for model in map(choose_signal_model, set(device_types)) if model
    ]


def load_signal_model(file_path: str) -> SignalModel:
    """Load signal model from persisted file.

    Models are assumed to have been written using bcipy.helpers.save.save_model
    function and should be serialized as pickled files. Note that reading
    pickled files is a potential security concern so only load from trusted
    directories."""

    with open(file_path, "rb") as signal_file:
        model = pickle.load(signal_file)
        log.info(f"Loading model {model}")
        return model


def choose_signal_model(device_type: str) -> Optional[SignalModel]:
    """Present a file dialog prompting the user to select a signal model for
    the given device.

    Parameters
    ----------
        device_type - ex. 'EEG' or 'Eyetracker'; this should correspond with
            the content_type of the DeviceSpec of the model.
    """

    file_path = ask_filename(file_types=f"*{SIGNAL_MODEL_FILE_SUFFIX}",
                             directory=preferences.signal_model_directory,
                             prompt=f"Select the {device_type} signal model")

    if file_path:
        # update preferences
        path = Path(file_path)
        preferences.signal_model_directory = str(path)
        return load_signal_model(str(path))
    return None


def choose_csv_file(filename: Optional[str] = None) -> Optional[str]:
    """GUI prompt to select a csv file from the file system.

    Parameters
    ----------
    - filename : optional filename to use; if provided the GUI is not shown.

    Returns
    -------
    file name of selected file; throws an exception if the file is not a csv.
    """
    if not filename:
        filename = ask_filename('*.csv')

    # get the last part of the path to determine file type
    file_name = filename.split('/')[-1]

    if 'csv' not in file_name:
        raise Exception(
            'File type unrecognized. Please use a supported csv type')

    return filename


def load_raw_data(filename: str) -> RawData:
    """Reads the data (.csv) file written by data acquisition.

    Parameters
    ----------
    - filename : path to the serialized data (csv file)

    Returns
    -------
    RawData object with data held in memory
    """
    return RawData.load(filename)


def load_txt_data() -> str:
    filename = ask_filename('*.txt')  # show dialog box and return the path
    file_name = filename.split('/')[-1]

    if 'txt' not in file_name:
        raise Exception(
            'File type unrecognized. Please use a supported text type')

    return filename


def load_users(data_save_loc: str) -> List[str]:
    """Load Users.

    Loads user directory names below experiments from the data path defined and returns them as a list.
    If the save data directory is not found, this method returns an empty list assuming no experiments
    have been run yet.
    """
    # build a saved users list, pull out the data save location from parameters
    saved_users: List[str] = []

    # check the directory is valid, if it is, set path as data save location
    if os.path.isdir(data_save_loc):
        path = data_save_loc

    # check the directory is valid after adding bcipy, if it is, set path as data save location
    elif os.path.isdir(f'{ROOT}/{data_save_loc}'):
        path = f'{ROOT}/{data_save_loc}'

    else:
        log.info(f'User save data location not found at [{data_save_loc}]! Returning empty user list.')
        return saved_users

    # grab all experiments in the directory and iterate over them to get the users
    experiments = fast_scandir(path, return_path=True)

    for experiment in experiments:
        users = fast_scandir(experiment, return_path=False)
        # If it is a new user, append it to the saved_user list
        for user in users:
            if user not in saved_users:
                saved_users.append(user)

    return saved_users


def fast_scandir(directory_name: str, return_path: bool = True) -> List[str]:
    """Fast Scan Directory.

    directory_name: name of the directory to be scanned
    return_path: whether or not to return the scanned directories as a relative path or name.
        False will return the directory name only.
    """
    if return_path:
        return [f.path for f in os.scandir(directory_name) if f.is_dir()]

    return [f.name for f in os.scandir(directory_name) if f.is_dir()]
