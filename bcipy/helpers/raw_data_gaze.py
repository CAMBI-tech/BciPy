import csv
from typing import List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd

from bcipy.config import DEFAULT_ENCODING

class RawGazeData:
    """Represents the raw eye gaze data format used by BciPy. Used primarily for loading
    a raw data file into memory."""

    def __init__(self,
                 daq_type: Optional[str] = None,
                 sample_rate: Optional[int] = None,
                 columns: Optional[List[str]] = None,
                 column_types: Optional[List[str]] = None):
        self.daq_type = daq_type
        self.sample_rate = sample_rate
        self.columns = columns or []
        # accept a custom column type definition or default to all eeg type
        self.column_types = column_types
        self._rows = []
        self._dataframe = None
    
    @classmethod
    def load(cls, filename: str):
        """Constructs a RawData object by deserializing the given file.
        All data will be read into memory. 

        Parameters
        ----------
        - filename : path to the csv file to read
        """
        return load(filename)
    
    @property
    def rows(self) -> List[List]:
        """Returns the data rows"""
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = value
        self._dataframe = None

    @property
    def channels(self) -> List[str]:
        """Compute the list of channels. Channels are the numeric columns
        excluding the timestamp column."""

        # Start data slice at 1 to remove the timestamp column.
        return list(self.numeric_data.columns[1:])
    
    @property
    def numeric_data(self) -> pd.DataFrame:
        """Data for columns with numeric data. This is usually comprised of the
        timestamp column and device channels, excluding string triggers."""
        return self.dataframe.select_dtypes(exclude=['object'])
    
    @property
    def channel_data(self):
        """Data for columns with numeric data, excluding the timestamp column."""
        numeric_data = self.numeric_data

        numeric_vals = numeric_data.values
        numeric_column_count = numeric_vals.shape[1]
        # Start data slice at 1 to remove the timestamp column.
        return numeric_vals[:, 1:numeric_column_count].transpose()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns a dataframe of the row data."""
        if self._dataframe is None:
            self._dataframe = pd.DataFrame(data=self.rows,
                                           columns=self.columns)
        return self._dataframe


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

