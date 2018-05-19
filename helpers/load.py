# -*- coding: utf-8 -*-
from tkinter import Tk
import numpy as np
import pandas as pd
from codecs import open as codecsopen
from json import load as jsonload
import pickle

from tkinter.filedialog import askopenfilename, askdirectory


def _cast_parameters(parameters):
    """Cast to Value.

    Take in a parameters json and coverts to a dictionary with type converted
        and extranous information removed
    """
    new_parameters = {}
    for key, value in parameters.items():
        new_parameters[key] = cast_value(value)

    return new_parameters


def cast_value(value):
    actual_value = str(value['value'])
    actual_type = value['type']

    try:
        if actual_type == 'int':
            new_value = int(actual_value)
        elif actual_type == 'float':
            new_value = float(actual_value)
        elif actual_type == 'bool':
            new_value = True if actual_type == 'true' else False
        elif actual_type == 'str':
            new_value = str(actual_value)
        else:
            raise ValueError('Unrecognized value type')

    except Exception:
        raise ValueError(f'Could not cast {actual_value} to {actual_type}')

    return new_value


def load_json_parameters(path, value_cast=False):
    # loads in json parameters and turns it into a dictionary
    try:
        with codecsopen(path, 'r', encoding='utf-8') as f:
            parameters = []
            try:
                parameters = jsonload(f)

                if value_cast:
                    parameters = _cast_parameters(parameters)
            except ValueError:
                raise ValueError(
                    "Parameters file is formatted incorrectly!")

        f.close()

    except IOError:
        raise IOError("Incorrect path to parameters given! Please try again.")

    return parameters


def load_experimental_data():
    # use python's internal gui to call file explorers and get the filename
    try:
        Tk().withdraw()  # we don't want a full GUI
        filename = askdirectory()  # show dialog box and return the path

    except Exception as error:
        raise error

    print("Loaded Experimental Data From: %s" % filename)
    return filename


def load_classifier(filename=None):
    # use python's internal gui to call file explorers and get the filename

    if not filename:
        try:
            Tk().withdraw()  # we don't want a full GUI
            filename = askopenfilename()  # show dialog box and return the path

        # except, raise error
        except Exception as error:
            raise error

    # load the classifier with pickle
    classifier = pickle.load(open(filename, 'rb'))

    return classifier


def load_csv_data(filename=None):
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


def read_data_csv(folder, dat_first_row=2, info_end_row=1):
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

    cols = list(dat_file.axes[1])
    channels = cols[2:len(cols)]

    temp = pd.DataFrame.as_matrix(dat_file)
    stamp_time = temp[:, 0]
    raw_dat = temp[:, 1:temp.shape[1]].transpose()

    dat_file_2 = pd.read_csv(folder, nrows=info_end_row)
    type_amp = list(dat_file_2.axes[1])[1]
    fs = np.array(dat_file_2)[0][1]

    return raw_dat, stamp_time, channels, type_amp, fs


def load_txt_data():
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
