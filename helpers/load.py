# -*- coding: utf-8 -*-

from Tkinter import Tk

from codecs import open as codecsopen
from json import load as jsonload

from tkFileDialog import askopenfilename, askdirectory


def load_json_parameters(path):
    # loads in json parameters and turns it into a dictionary
    try:
        with codecsopen(path, 'r', encoding='utf-8') as f:
            parameters = []
            try:
                parameters = jsonload(f)
            except ValueError as e:
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


def load_classifier():

    return


def load_csv_data():
    try:
        Tk().withdraw()  # we don't want a full GUI
        filename = askopenfilename()  # show dialog box and return the path

    except Exception as error:
        raise error

    # get the last part of the path to determine file type
    eeg_data_file_name = filename.split('/')[-1]

    if 'csv' not in eeg_data_file_name:
        raise Exception(
            'File type unrecognized. Please use a supported eeg type')

    # give the user some insight into what's happening
    print("Loaded EEG Data From: %s" % filename)

    return filename


def load_txt_data():

    try:
        Tk().withdraw()  # we don't want a full GUI
        filename = askopenfilename()  # show dialog box and return the path
    except Exception as error:
        raise error

    trigger_file_name = filename.split('/')[-1]

    if 'txt' not in trigger_file_name:
        raise Exception(
            'File type unrecognized. Please use a supported trigger type')

    # give the user some insight into what's happening
    print("Loaded Trigger Data From: %s" % filename)

    return filename
