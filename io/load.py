from Tkinter import Tk

from codecs import open as codecsopen
from json import load as jsonload

from tkFileDialog import askopenfilename


def load_json_parameters(path):
    # loads in json parameters and turns it into a dictionary
    try:
        with codecsopen(path, 'r', encoding='utf-8') as f:
            parameters = []
            try:
                parameters = jsonload(f)
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
        filename = askopenfilename()  # show dialog box and return the path
        print("Loaded Experimental Data From: %s" % filename)
    except:
        pass

    return filename


def load_classifier():

    return


def read_csv_data():

    return
