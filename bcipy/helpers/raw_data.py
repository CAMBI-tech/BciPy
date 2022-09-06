"""Functionality for reading and writing raw signal data."""
import csv
from typing import List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd

from bcipy.config import DEFAULT_ENCODING
from bcipy.signal.generator.generator import gen_random_data
from bcipy.signal.process import Composition

TIMESTAMP_COLUMN = 'timestamp'


class RawData:
    """Represents the raw data format used by BciPy. Used primarily for loading
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
        All data will be read into memory. If you want to lazily read data one
        record at a time, use a RawDataReader.

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

    def by_channel(self, transform: Optional[Composition] = None) -> np.ndarray:
        """Data organized by channel.

        Optionally, it can apply a BciPy Composition to the data before returning using the transform arg.
        This will apply n tranformations to the data before returning. For an example Composition with
        EEG preprocessing, see bcipy.signal.get_default_transform().

        Returns
        ----------
        data: C x N numpy array with samples where C is the number of channels and N
        is number of time samples
        fs: resulting sample rate if any transformations applied"""

        data = self.channel_data
        fs = self.sample_rate

        if transform:
            data, fs = self.apply_transform(data, transform)

        return data, fs

    def by_channel_map(
            self,
            channel_map: List[int],
            transform: Optional[Composition] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """By Channel Map.

        Returns channels with columns removed if index in list (channel_map) is zero. The channel map must align
        with the numeric channels read in as self.channels. We assume most trigger or other string columns are
        removed, however some are numeric trigger columns from devices that will require filtering before returning
        data. Other cases could be dropping bad channels before running further analyses.

        Optionally, it can apply a BciPy Composition to the data before returning using the transform arg.
        This will apply n tranformations to the data before returning. For an example Composition with
        EEG preprocessing, see bcipy.signal.get_default_transform().
        """
        data, fs = self.by_channel(transform)
        channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
        data = np.delete(data, channels_to_remove, axis=0)
        channels = np.delete(self.channels, channels_to_remove, axis=0).tolist()

        return data, channels, fs

    def apply_transform(self, data: np.ndarray, transform: Composition) -> Tuple[np.ndarray, float]:
        """Apply Transform.

        Using data provided as an np.ndarray, call the Composition with self.sample_rate to apply
            transformations to the data. This will return the transformed data and resulting sample rate.
        """
        return transform(data, self.sample_rate)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns a dataframe of the row data."""
        if self._dataframe is None:
            self._dataframe = pd.DataFrame(data=self.rows,
                                           columns=self.columns)
        return self._dataframe

    @property
    def total_seconds(self) -> float:
        """Total recorded seconds, defined as the diff between the first and
        last timestamp."""
        frame = self.dataframe
        col = 'lsl_timestamp'
        return frame.iloc[-1][col] - frame.iloc[0][col]

    @property
    def total_samples(self) -> int:
        """Number of samples recorded."""
        return int(self.dataframe.iloc[-1]['timestamp'])

    def query(self,
              start: Optional[float] = None,
              stop: Optional[float] = None,
              column: str = 'lsl_timestamp') -> pd.DataFrame:
        """Query for a subset of data.

        Pararameters
        ------------
            start - find records greater than or equal to this value
            stop - find records less than or equal to this value
            column - column to compare to the given start and stop values

        Returns
        -------
        Dataframe for the given slice of data
        """
        dataframe = self.dataframe
        start = start or dataframe.iloc[0][column]
        stop = stop or dataframe.iloc[-1][column]
        mask = (dataframe[column] >= start) & (dataframe[column] <= stop)
        return dataframe[mask]

    def append(self, row: List):
        """Append the given row of data.

        Parameters
        ----------
        - row : row of data
        """
        assert len(row) == len(self.columns), "Wrong number of columns"
        self._rows.append(row)
        self._dataframe = None


def maybe_float(val):
    """Attempt to convert the given value to float. If conversion fails return
    as is."""
    try:
        return float(val)
    except ValueError:
        return val


class RawDataReader:
    """Lazily reads raw data from a file. Intended to be used as a ContextManager
    using Python's `with` keyword.

    Example usage:

    ```
    with RawDataReader(path) as reader:
        print(f"Data from ${reader.daq_type}")
        print(reader.columns)
        for row in reader:
            print(row)
    ```

    Parameters
    ----------
    - file_path : path to the csv file
    - convert_data : if True attempts to convert data values to floats;
    default is False
    """

    def __init__(self, file_path: str, convert_data: bool = False):
        self.file_path = file_path
        self._file_obj = None
        self.convert_data = convert_data

    def __enter__(self):
        self._file_obj = open(self.file_path, mode="r", encoding=DEFAULT_ENCODING)
        self.daq_type, self.sample_rate = read_metadata(self._file_obj)
        self._reader = csv.reader(self._file_obj)
        self.columns = next(self._reader)
        return self

    def __exit__(self, *args, **kwargs):
        """Exit the context manager. Close resources"""
        self._file_obj.close()

    def __iter__(self):
        return self

    def __next__(self):
        assert self._reader, "Reader must be initialized"
        row = next(self._reader)
        if self.convert_data:
            return list(map(maybe_float, row))
        return row


class RawDataWriter:
    """Writes a raw data file one row at a time without storing the records
    in memory. Intended to be used as a ContextManager using Python's `with`
    keyword.

    Example usage:

    ```
    with RawDataWriter(path, daq_type, sample_rate, columns) as writer:
        for row in data:
            writer.writerow(row)
    ```

    Parameters
    ----------
    - file_path : path to the csv file
    - daq_type : name of device
    - sample_rate : sample frequency in Hz
    - columns : list of column names
    """

    def __init__(self, file_path: str, daq_type: str, sample_rate: float,
                 columns: List[str]):
        self.file_path = file_path
        self.daq_type = daq_type
        self.sample_rate = sample_rate
        self.columns = columns
        self._file_obj = None

    def __enter__(self):
        """Enter the context manager. Initializes the underlying data file."""
        self._file_obj = open(self.file_path,
                              mode='w',
                              encoding=DEFAULT_ENCODING,
                              newline='')
        # write metadata
        self._file_obj.write(f"daq_type,{self.daq_type}\n")
        self._file_obj.write(f"sample_rate,{self.sample_rate}\n")

        # If flush is missing the previous content may be appended at the end.
        self._file_obj.flush()

        self._csv_writer = csv.writer(self._file_obj, delimiter=',')
        self._csv_writer.writerow(self.columns)

        return self

    def __exit__(self, *args, **kwargs):
        """Exit the context manager. Close resources"""
        self._file_obj.close()

    def writerow(self, row: List) -> None:
        assert self._csv_writer, "Writer must be initialized"
        assert len(row) == len(self.columns), "Wrong number of columns"
        self._csv_writer.writerow(row)

    def writerows(self, rows: List[List]) -> None:
        for row in rows:
            self.writerow(row)


def load(filename: str) -> RawData:
    """Reads the file at the given path and initializes a RawData object.
    All data will be read into memory. If you want to lazily read data one
    record at a time, use a RawDataReader.

    Parameters
    ----------
    - filename : path to the csv file to read
    """
    # Loading all data from a csv is faster using pandas than using the
    # RawDataReader.
    with open(filename, mode='r', encoding=DEFAULT_ENCODING) as file_obj:
        daq_type, sample_rate = read_metadata(file_obj)
        dataframe = pd.read_csv(file_obj)
        data = RawData(daq_type, sample_rate, list(dataframe.columns))
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


def write(data: RawData, filename: str):
    """Write the given raw data file.

    Parameters
    ----------
    - data : raw data object to write
    - filename : path to destination file.
    """
    with RawDataWriter(filename, data.daq_type, data.sample_rate,
                       data.columns) as writer:
        for row in data.rows:
            writer.writerow(row)


def settings(filename: str) -> Tuple[str, float, List[str]]:
    """Read the daq settings from the given data file

    Parameters
    ----------
    - filename : path to the raw data file (csv)

    Returns
    -------
    tuple of (acquisition type, sample_rate, columns)
    """

    with RawDataReader(filename) as reader:
        return reader.daq_type, reader.sample_rate, reader.columns


def sample_data(rows: int = 1000,
                ch_names: List[str] = ['ch1', 'ch2', 'ch3'],
                daq_type: str = 'SampleDevice',
                sample_rate: float = 256.0,
                triggers: List[Tuple[float, str]] = []) -> RawData:
    """Creates sample data to be written as a raw_data.csv file. The resulting data has
    a column for the timestamp, one for each channel, and a TRG column.

    - rows : number of sample rows to generate
    - ch_names : list of channel names
    - daq_type : metadata for the device name
    - sample_rate : device sample rate in hz
    - triggers : List of (timestamp, trigger_value) tuples to be inserted
    in the data.
    """
    channels = [name for name in ch_names if name != 'TRG']
    columns = [TIMESTAMP_COLUMN] + channels + ['TRG']
    trg_times = dict(triggers)

    data = RawData(daq_type, sample_rate, columns)

    for i in range(rows):
        timestamp = i + 1
        channel_data = gen_random_data(low=-1000,
                                       high=1000,
                                       channel_count=len(channels))
        trg = trg_times.get(timestamp, '0.0')
        data.append([timestamp] + channel_data + [trg])

    return data
