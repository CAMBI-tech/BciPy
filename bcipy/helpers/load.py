import logging
import json
import os
from pathlib import Path
from shutil import copyfile
from time import localtime, strftime
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename

from typing import List, Dict, Any

import numpy as np
import pandas as pd

from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH, Parameters
from bcipy.helpers.system_utils import DEFAULT_EXPERIMENT_PATH, DEFAULT_FIELD_PATH, EXPERIMENT_FILENAME, FIELD_FILENAME
from bcipy.helpers.exceptions import BciPyCoreException, InvalidExperimentException
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


def load_experiments(path: str = f'{DEFAULT_EXPERIMENT_PATH}{EXPERIMENT_FILENAME}') -> dict:
    """Load Experiments.

    PARAMETERS
    ----------
    :param: path: string path to the experiments file.

    Returns
    -------
        A dictionary of experiments, with the following format:
            { name: { fields : {name: '', required: bool}, summary: '' } }

    """
    with open(path, 'r') as json_file:
        return json.load(json_file)


def extract_mode(bcipy_data_directory: str) -> str:
    """Extract Mode.

    This method extracts the task mode from a BciPy data save directory. This is important for
        trigger conversions and extracting targeteness.

    *note*: this is not compatiable with older versions of BciPy (pre 1.5.0) where
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
    elif 'free_spell' in directory:
        return 'free_spell'
    raise BciPyCoreException(f'No valid mode could be extracted from [{directory}]')


def load_fields(path: str = f'{DEFAULT_FIELD_PATH}{FIELD_FILENAME}') -> dict:
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
    with open(path, 'r') as json_file:
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
    # use python's internal gui to call file explorers and get the filename
    try:
        Tk().withdraw()  # we don't want a full GUI
        filename = askdirectory()  # show dialog box and return the path

    except Exception as error:
        raise error

    log.debug("Loaded Experimental Data From: %s" % filename)
    return filename


def load_signal_model(model_class: SignalModel, model_kwargs: Dict[str, Any], filename: str = None):
    """Construct the specified model and load pretrained parameters.

    Args:
        model_class (SignalModel, optional): Model class to construct.
        model_kwargs (dict, optional): Keyword arguments for constructing model.
        filename (str, optional): Location of pretrained model parameters.

    Returns:
        SignalModel: Model after loading pretrained parameters.
    """
    # use python's internal gui to call file explorers and get the filename

    if not filename:
        try:
            Tk().withdraw()  # we don't want a full GUI
            filename = askopenfilename()  # show dialog box and return the path

        # except, raise error
        except Exception as error:
            raise error

    # load the signal_model with pickle
    signal_model = model_class(**model_kwargs)
    with open(filename, "rb") as f:
        signal_model.load(f)

    return signal_model, filename


def load_csv_data(filename: str = None) -> str:
    if not filename:
        try:
            Tk().withdraw()  # we don't want a full GUI
            filename = askopenfilename()  # show dialog box and return the path

        except Exception as error:
            raise error

    # get the last part of the path to determine file type
    file_name = filename.split('/')[-1]

    if 'csv' not in file_name:
        raise Exception(
            'File type unrecognized. Please use a supported csv type')

    return filename


def read_data_csv(folder: str, dat_first_row: int = 2,
                  info_end_row: int = 1) -> tuple:
    """ Reads the data (.csv) provided by the data acquisition
        Arg:
            folder(str): file location for the data
            dat_first_row(int): row with channel names
            info_end_row(int): final row related with daq. info
                where first row idx is '0'
        Return:
            raw_dat(ndarray[float]): C x N numpy array with samples
                where C is number of channels N is number of time samples
            channels(list[str]): channels used in DAQ
            stamp_time(ndarray[float]): time stamps for each sample
            type_amp(str): type of the device used for DAQ
            fs(int): sampling frequency
    """
    dat_file = pd.read_csv(folder, skiprows=dat_first_row)

    # Remove object columns (ex. BCI_Stimulus_Marker column)
    # TODO: might be better in use:
    # dat_file.select_dtypes(include=['float64'])
    numeric_dat_file = dat_file.select_dtypes(exclude=['object'])
    channels = list(numeric_dat_file.columns[1:])  # without timestamp column

    temp = numeric_dat_file.values
    stamp_time = temp[:, 0]
    raw_dat = temp[:, 1:temp.shape[1]].transpose()

    dat_file_2 = pd.read_csv(folder, nrows=info_end_row)
    type_amp = list(dat_file_2.axes[1])[1]
    fs = np.array(dat_file_2)[0][1]

    return raw_dat, stamp_time, channels, type_amp, fs


def load_txt_data() -> str:
    try:
        Tk().withdraw()  # we don't want a full GUI
        filename = askopenfilename()  # show dialog box and return the path
    except Exception as error:
        raise error

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
    elif os.path.isdir(f'bcipy/{data_save_loc}'):
        path = f'bcipy/{data_save_loc}'

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
