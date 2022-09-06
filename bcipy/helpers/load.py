import json
import logging
import os
from pathlib import Path
from shutil import copyfile
from time import localtime, strftime
from typing import Any, Dict, List, Tuple

from bcipy.config import (
    ROOT,
    DEFAULT_ENCODING,
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_PARAMETERS_PATH,
    DEFAULT_FIELD_PATH,
    EXPERIMENT_FILENAME,
    FIELD_FILENAME)
from bcipy.gui.file_dialog import ask_directory, ask_filename
from bcipy.helpers.exceptions import (BciPyCoreException,
                                      InvalidExperimentException)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.raw_data import RawData
from bcipy.signal.model import SignalModel

log = logging.getLogger(__name__)


def copy_parameters(path: str = DEFAULT_PARAMETERS_PATH,
                    destination: str = None) -> str:
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
    log.debug("Loaded Experimental Data From: %s" % filename)
    return filename


def load_signal_model(model_class: SignalModel,
                      model_kwargs: Dict[str, Any], filename: str = None) -> Tuple[SignalModel, str]:
    """Construct the specified model and load pretrained parameters.

    Args:
        model_class (SignalModel, optional): Model class to construct.
        model_kwargs (dict, optional): Keyword arguments for constructing model.
        filename (str, optional): Location of pretrained model parameters.

    Returns:
        SignalModel: Model after loading pretrained parameters.
    """
    # use python's internal gui to call file explorers and get the filename
    if not filename or Path(filename).is_dir():
        filename = ask_filename('*.pkl', filename)

    # load the signal_model with pickle
    signal_model = model_class(**model_kwargs)
    signal_model.load(filename)

    log.info(f'Loaded signal model from {filename}')

    return signal_model, filename


def choose_csv_file(filename: str = None) -> str:
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


def load_users(data_save_loc) -> List[str]:
    """Load Users.

    Loads user directory names below experiments from the data path defined and returns them as a list.
    If the save data directory is not found, this method returns an empty list assuming no experiments
    have been run yet.
    """
    # build a saved users list, pull out the data save location from parameters
    saved_users = []

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
