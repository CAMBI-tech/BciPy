"""Functionality for reading and writing raw signal data."""
import csv
from io import TextIOWrapper
from typing import Any, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from bcipy.config import DEFAULT_ENCODING
from bcipy.exceptions import BciPyCoreException
from bcipy.signal.generator.generator import gen_random_data
from bcipy.signal.process import Composition

TIMESTAMP_COLUMN = 'timestamp'


class RawData:
    """Represents the raw data format used by BciPy.

    This class is used primarily for loading a raw data file into memory. It provides
    methods to access and manipulate the data in various formats.

    Attributes:
        daq_type (str): Type of data acquisition device.
        sample_rate (int): Sample rate in Hz.
        columns (List[str]): List of column names in the data.
    """

    def __init__(self, daq_type: str, sample_rate: int, columns: List[str]) -> None:
        """Initialize RawData.

        Args:
            daq_type (str): Type of data acquisition device.
            sample_rate (int): Sample rate in Hz.
            columns (List[str]): List of column names in the data.
        """
        self.daq_type = daq_type
        self.sample_rate = sample_rate
        self.columns = columns

        self._dataframe: Optional[pd.DataFrame] = None
        self._rows: List[Any] = []

    @classmethod
    def load(cls, filename: str) -> 'RawData':
        """Constructs a RawData object by deserializing the given file.

        All data will be read into memory. If you want to lazily read data one
        record at a time, use a RawDataReader.

        Args:
            filename (str): Path to the csv file to read.

        Returns:
            RawData: A new RawData instance containing the loaded data.
        """
        return load(filename)

    @property
    def rows(self) -> List[List]:
        """Returns the data rows.

        Returns:
            List[List]: List of data rows.
        """
        return self._rows

    @rows.setter
    def rows(self, value: Any) -> None:
        """Sets the data rows and invalidates the cached dataframe.

        Args:
            value (Any): New data rows to set.
        """
        self._rows = value
        self._dataframe = None

    @property
    def channels(self) -> List[str]:
        """Compute the list of channels.

        Channels are the numeric columns excluding the timestamp column.

        Returns:
            List[str]: List of channel names.
        """
        # Start data slice at 1 to remove the timestamp column.
        return list(self.numeric_data.columns[1:])

    @property
    def numeric_data(self) -> pd.DataFrame:
        """Data for columns with numeric data.

        This is usually comprised of the timestamp column and device channels,
        excluding string triggers.

        Returns:
            pd.DataFrame: DataFrame containing only numeric columns.
        """
        return self.dataframe.select_dtypes(exclude=['object'])

    @property
    def channel_data(self) -> np.ndarray:
        """Data for columns with numeric data, excluding the timestamp column.

        Returns:
            np.ndarray: Array of channel data with shape (channels, samples).
        """
        numeric_data = self.numeric_data

        numeric_vals = numeric_data.values
        numeric_column_count = numeric_vals.shape[1]
        # Start data slice at 1 to remove the timestamp column.
        return numeric_vals[:, 1:numeric_column_count].transpose()

    def by_channel(self, transform: Optional[Composition] = None) -> Tuple[np.ndarray, int]:
        """Data organized by channel.

        Optionally, it can apply a BciPy Composition to the data before returning using the transform arg.
        This will apply n tranformations to the data before returning. For an example Composition with
        EEG preprocessing, see bcipy.signal.get_default_transform().

        Args:
            transform (Optional[Composition]): Optional transformation to apply to the data.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing:
                - data: C x N numpy array with samples where C is the number of channels and N
                  is number of time samples
                - fs: resulting sample rate if any transformations applied
        """
        data = self.channel_data
        fs = self.sample_rate

        if transform:
            data, fs = self.apply_transform(data, transform)

        return data, fs

    def by_channel_map(
            self,
            channel_map: List[int],
            transform: Optional[Composition] = None) -> Tuple[np.ndarray, List[str], int]:
        """Returns channels with columns removed if index in list (channel_map) is zero.

        The channel map must align with the numeric channels read in as self.channels.
        We assume most trigger or other string columns are removed, however some are
        numeric trigger columns from devices that will require filtering before returning
        data. Other cases could be dropping bad channels before running further analyses.

        Args:
            channel_map (List[int]): List of 1s and 0s indicating which channels to keep.
            transform (Optional[Composition]): Optional transformation to apply to the data.

        Returns:
            Tuple[np.ndarray, List[str], int]: A tuple containing:
                - data: Array of channel data with shape (channels, samples)
                - channels: List of channel names
                - fs: Sample rate
        """
        data, fs = self.by_channel(transform)
        channels_to_remove = [idx for idx,
                              value in enumerate(channel_map) if value == 0]
        data = np.delete(data, channels_to_remove, axis=0)
        channels: List[str] = np.delete(
            self.channels, channels_to_remove, axis=0).tolist()

        return data, channels, fs

    def apply_transform(self, data: np.ndarray, transform: Composition) -> Tuple[np.ndarray, int]:
        """Apply Transform.

        Using data provided as an np.ndarray, call the Composition with self.sample_rate to apply
        transformations to the data. This will return the transformed data and resulting sample rate.

        Args:
            data (np.ndarray): Input data to transform.
            transform (Composition): Transformation to apply.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing:
                - data: Transformed data
                - fs: Resulting sample rate
        """
        return transform(data, self.sample_rate)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns a dataframe of the row data.

        Returns:
            pd.DataFrame: DataFrame containing all data.
        """
        if self._dataframe is None:
            self._dataframe = pd.DataFrame(data=self.rows,
                                           columns=self.columns)
        return self._dataframe

    @property
    def total_seconds(self) -> float:
        """Total recorded seconds.

        Defined as the diff between the first and last timestamp.

        Returns:
            float: Total duration in seconds.
        """
        frame = self.dataframe
        col = 'lsl_timestamp'
        return frame.iloc[-1][col] - frame.iloc[0][col]

    @property
    def total_samples(self) -> int:
        """Number of samples recorded.

        Returns:
            int: Total number of samples.
        """
        return int(self.dataframe.iloc[-1]['timestamp'])

    def query(self,
              start: Optional[float] = None,
              stop: Optional[float] = None,
              column: str = 'lsl_timestamp') -> pd.DataFrame:
        """Query for a subset of data.

        Args:
            start (Optional[float]): Find records greater than or equal to this value.
            stop (Optional[float]): Find records less than or equal to this value.
            column (str): Column to compare to the given start and stop values.

        Returns:
            pd.DataFrame: DataFrame for the given slice of data.
        """
        dataframe = self.dataframe
        start = start or dataframe.iloc[0][column]
        stop = stop or dataframe.iloc[-1][column]
        mask = (dataframe[column] >= start) & (dataframe[column] <= stop)
        return dataframe[mask]

    def append(self, row: list) -> None:
        """Append the given row of data.

        Args:
            row (list): Row of data to append.
        """
        assert len(row) == len(self.columns), "Wrong number of columns"
        self._rows.append(row)
        self._dataframe = None

    def __str__(self) -> str:
        """String representation of RawData.

        Returns:
            str: String representation.
        """
        return f"RawData({self.daq_type})"

    def __repr__(self) -> str:
        """String representation of RawData.

        Returns:
            str: String representation.
        """
        return f"RawData({self.daq_type})"


def maybe_float(val: Any) -> Union[float, Any]:
    """Attempt to convert the given value to float.

    If conversion fails return the value as is.

    Args:
        val (Any): Value to convert to float.

    Returns:
        Union[float, Any]: Float if conversion succeeds, original value otherwise.
    """
    try:
        return float(val)
    except ValueError:
        return val


class RawDataReader:
    """Lazily reads raw data from a file.

    Intended to be used as a ContextManager using Python's `with` keyword.

    Example usage:
        ```
        with RawDataReader(path) as reader:
            print(f"Data from ${reader.daq_type}")
            print(reader.columns)
            for row in reader:
                print(row)
        ```

    Attributes:
        file_path (str): Path to the csv file.
        convert_data (bool): If True attempts to convert data values to floats.
        daq_type (str): Type of data acquisition device.
        sample_rate (int): Sample rate in Hz.
        columns (List[str]): List of column names.
    """
    _file_obj: TextIOWrapper

    def __init__(self, file_path: str, convert_data: bool = False):
        """Initialize RawDataReader.

        Args:
            file_path (str): Path to the csv file.
            convert_data (bool, optional): If True attempts to convert data values to floats.
                Defaults to False.
        """
        self.file_path = file_path
        self.convert_data = convert_data

    def __enter__(self) -> 'RawDataReader':
        """Enter the context manager.

        Returns:
            RawDataReader: Self.
        """
        self._file_obj = open(self.file_path, mode="r",
                              encoding=DEFAULT_ENCODING)
        self.daq_type, self.sample_rate = read_metadata(self._file_obj)
        self._reader = csv.reader(self._file_obj)
        self.columns = next(self._reader)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Exit the context manager. Close resources."""
        if self._file_obj:
            self._file_obj.close()

    def __iter__(self) -> 'RawDataReader':
        """Return self as iterator.

        Returns:
            RawDataReader: Self.
        """
        return self

    def __next__(self) -> List[Any]:
        """Get next row of data.

        Returns:
            List[Any]: Next row of data.

        Raises:
            AssertionError: If reader is not initialized.
        """
        assert self._reader, "Reader must be initialized"
        row = next(self._reader)
        if self.convert_data:
            return list(map(maybe_float, row))
        return row


class RawDataWriter:
    """Writes a raw data file one row at a time without storing the records in memory.

    Intended to be used as a ContextManager using Python's `with` keyword.

    Example usage:
        ```
        with RawDataWriter(path, daq_type, sample_rate, columns) as writer:
            for row in data:
                writer.writerow(row)
        ```

    Attributes:
        file_path (str): Path to the csv file.
        daq_type (str): Name of device.
        sample_rate (float): Sample frequency in Hz.
        columns (List[str]): List of column names.
    """
    _file_obj: TextIOWrapper

    def __init__(self, file_path: str, daq_type: str, sample_rate: float,
                 columns: List[str]) -> None:
        """Initialize RawDataWriter.

        Args:
            file_path (str): Path to the csv file.
            daq_type (str): Name of device.
            sample_rate (float): Sample frequency in Hz.
            columns (List[str]): List of column names.
        """
        self.file_path = file_path
        self.daq_type = daq_type
        self.sample_rate = sample_rate
        self.columns = columns

    def __enter__(self) -> 'RawDataWriter':
        """Enter the context manager. Initializes the underlying data file.

        Returns:
            RawDataWriter: Self.
        """
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

    def __exit__(self, *args, **kwargs) -> None:
        """Exit the context manager. Close resources."""
        self._file_obj.close()

    def writerow(self, row: List) -> None:
        """Write a single row of data.

        Args:
            row (List): Row of data to write.

        Raises:
            AssertionError: If writer is not initialized or row has wrong number of columns.
        """
        assert self._csv_writer, "Writer must be initialized"
        assert len(row) == len(self.columns), "Wrong number of columns"
        self._csv_writer.writerow(row)

    def writerows(self, rows: List[List]) -> None:
        """Write multiple rows of data.

        Args:
            rows (List[List]): Rows of data to write.
        """
        for row in rows:
            self.writerow(row)


def load(filename: str) -> RawData:
    """Reads the file at the given path and initializes a RawData object.

    All data will be read into memory. If you want to lazily read data one
    record at a time, use a RawDataReader.

    Args:
        filename (str): Path to the csv file to read.

    Returns:
        RawData: A new RawData instance containing the loaded data.

    Raises:
        BciPyCoreException: If the file is not found.
    """
    # Loading all data from a csv is faster using pandas than using the
    # RawDataReader.
    try:
        with open(filename, mode='r', encoding=DEFAULT_ENCODING) as file_obj:
            daq_type, sample_rate = read_metadata(file_obj)
            dataframe = pd.read_csv(file_obj)
            data = RawData(daq_type, sample_rate, list(dataframe.columns))
            data.rows = dataframe.values.tolist()
            return data
    except FileNotFoundError:
        raise BciPyCoreException(
            f"\nError loading BciPy RawData. Valid data not found at: {filename}")


def read_metadata(file_obj: TextIO) -> Tuple[str, int]:
    """Reads the metadata from an open raw data file and returns the result as a tuple.

    Increments the reader.

    Args:
        file_obj (TextIO): Open TextIO object.

    Returns:
        Tuple[str, int]: Tuple of (daq_type, sample_rate).
    """
    daq_type = next(file_obj).strip().split(',')[1]
    sample_rate = int(float(next(file_obj).strip().split(",")[1]))
    return daq_type, sample_rate


def write(data: RawData, filename: str) -> None:
    """Write the given raw data file.

    Args:
        data (RawData): Raw data object to write.
        filename (str): Path to destination file.
    """
    with RawDataWriter(filename, data.daq_type, data.sample_rate,
                       data.columns) as writer:
        for row in data.rows:
            writer.writerow(row)


def settings(filename: str) -> Tuple[str, float, List[str]]:
    """Read the daq settings from the given data file.

    Args:
        filename (str): Path to the raw data file (csv).

    Returns:
        Tuple[str, float, List[str]]: Tuple of (acquisition type, sample_rate, columns).
    """
    with RawDataReader(filename) as reader:
        return reader.daq_type, reader.sample_rate, reader.columns


def sample_data(rows: int = 1000,
                ch_names: List[str] = ['ch1', 'ch2', 'ch3'],
                daq_type: str = 'SampleDevice',
                sample_rate: int = 256,
                triggers: List[Tuple[float, str]] = []) -> RawData:
    """Creates sample data to be written as a raw_data.csv file.

    The resulting data has a column for the timestamp, one for each channel,
    and a TRG column.

    Args:
        rows (int, optional): Number of sample rows to generate. Defaults to 1000.
        ch_names (List[str], optional): List of channel names. Defaults to ['ch1', 'ch2', 'ch3'].
        daq_type (str, optional): Metadata for the device name. Defaults to 'SampleDevice'.
        sample_rate (int, optional): Device sample rate in hz. Defaults to 256.
        triggers (List[Tuple[float, str]], optional): List of (timestamp, trigger_value) tuples
            to be inserted in the data. Defaults to [].

    Returns:
        RawData: A new RawData instance containing the sample data.
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


def get_1020_channels() -> List[str]:
    """Returns the standard 10-20 channel names.

    Note: The 10-20 system is a standard for EEG electrode placement. The following
    is not a complete list of all possible channels, but the most common ones used
    in BCI research. This excludes the reference and ground channels.

    Returns:
        List[str]: List of channel names.
    """
    return [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
        'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'PO7', 'PO3', 'POz',
        'PO4', 'PO8', 'Oz', 'A1', 'A2',
    ]


def get_1020_channel_map(channels_name: List[str]) -> List[int]:
    """Returns a list of 1s and 0s indicating if the channel name is in the 10-20 system.

    Args:
        channels_name (List[str]): List of channel names.

    Returns:
        List[int]: List of 1s and 0s indicating if the channel name is in the 10-20 system.
    """
    valid_channels = get_1020_channels()
    return [1 if name in valid_channels else 0 for name in channels_name]
