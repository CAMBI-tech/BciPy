import csv
from typing import List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd

from bcipy.config import DEFAULT_ENCODING


def load(filename: str) -> RawGazeData:
    """Reads the file at the given path and initializes a RawData object.
    All data will be read into memory. 

    Parameters
    ----------
    - filename : path to the csv file to read
    """
    # Loading all data from a csv is faster using pandas than using the
    # RawDataReader.
    with open(filename, mode='r', encoding=DEFAULT_ENCODING) as file_obj:
        daq_type, sample_rate = read_metadata(file_obj)
        dataframe = pd.read_csv(file_obj)
        data = RawGazeData(daq_type, sample_rate, list(dataframe.columns))
        data.rows = dataframe.values.tolist()
        return data
    
def read_metadata(file_obj: TextIO) -> Tuple[str, float]:
    """Reads the metadata from an open raw data file and retuns the result as
    a tuple. Increments the reader.

    Parameters
    ----------
    - file_obj : open TextIO object

    Returns
    -------
    tuple of daq_type, sample_rate
    """
    daq_type = next(file_obj).strip().split(',')[1]
    sample_rate = float(next(file_obj).strip().split(",")[1])
    return daq_type, sample_rate

