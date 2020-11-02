import logging
import pickle
from pathlib import Path
from shutil import copyfile
from time import localtime, strftime
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename

import numpy as np
import pandas as pd

from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH, Parameters

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


def load_signal_model(filename: str = None):
    # use python's internal gui to call file explorers and get the filename

    if not filename:
        try:
            Tk().withdraw()  # we don't want a full GUI
            filename = askopenfilename()  # show dialog box and return the path

        # except, raise error
        except Exception as error:
            raise error

    # load the signal_model with pickle
    signal_model = pickle.load(open(filename, 'rb'))

    return (signal_model, filename)


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
