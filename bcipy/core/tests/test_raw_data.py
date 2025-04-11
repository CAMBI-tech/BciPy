"""Tests for BciPy raw data format."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mockito import any, mock, when, verify, unstub

from bcipy.exceptions import BciPyCoreException
from bcipy.core.raw_data import (RawData, RawDataReader, RawDataWriter,
                                 load, sample_data, settings, write,
                                 get_1020_channel_map, get_1020_channels)


class TestRawData(unittest.TestCase):
    """Tests for raw_data format"""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        self.data_dir = f"{os.path.dirname(__file__)}/resources/"
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir, 'test_raw_data.csv')
        self.daq_type = 'Test-Device'
        self.sample_rate = 300
        self.columns = ['timestamp', 'ch1', 'ch2', 'ch3']
        self.row1 = [1, 1.0, 2.0, 3.0]
        self.row2 = [2, 4.0, 5.0, 6.0]
        self.row3 = [3, 7.0, 8.0, 9.0]

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)
        unstub()

    def _write_raw_data(self, include_rows=False) -> RawData:
        """Helper function to write a sample raw data file to disk using the
        settings from setUp.

        Parameters
        ----------
        - include_rows : if True adds data, otherwise just writes the metadata
        and columns.
        """
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=self.columns)
        if include_rows:
            data.append(self.row1)
            data.append(self.row2)
            data.append(self.row3)

        write(data, self.path)

    def test_init(self):
        """Test that a RawData structure can be initialized"""
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=self.columns)
        self.assertEqual(data.daq_type, self.daq_type)
        self.assertEqual(data.sample_rate, self.sample_rate)
        self.assertEqual(data.columns, self.columns)

    def test_write_with_no_data(self):
        """Test that raw data can be persisted to disk."""
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=self.columns)

        self.assertFalse(self.path.exists())
        write(data, self.path)
        self.assertTrue(self.path.exists())

    def test_load_with_no_data(self):
        """Test that the raw data format can be read by the module."""
        self._write_raw_data()
        data = load(self.path)

        self.assertEqual(data.daq_type, self.daq_type)
        self.assertEqual(data.sample_rate, self.sample_rate)
        self.assertEqual(data.columns, self.columns)
        self.assertEqual(0, len(data.rows))

    def test_load_with_data(self):
        """Test that data can be loaded from a file."""
        self._write_raw_data(include_rows=True)
        loaded_data = load(self.path)

        self.assertEqual(loaded_data.daq_type, self.daq_type)
        self.assertEqual(loaded_data.sample_rate, self.sample_rate)
        self.assertEqual(loaded_data.columns, self.columns)

        self.assertEqual(len(loaded_data.rows), 3)
        self.assertEqual(loaded_data.rows[0], self.row1)
        self.assertEqual(loaded_data.rows[1], self.row2)
        self.assertEqual(loaded_data.rows[2], self.row3)

    def test_load_with_invalid_filepath(self):
        """Test that an exception is raised when an invalid file is loaded."""
        with self.assertRaises(BciPyCoreException):
            path = "invalid/path/to/file.csv"
            load(path)

    def test_deserialization(self):
        """Test that the load function can be accessed through a class
        constructor."""
        self._write_raw_data(include_rows=True)

        loaded_data = RawData.load(self.path)

        self.assertEqual(loaded_data.daq_type, self.daq_type)
        self.assertEqual(loaded_data.sample_rate, self.sample_rate)
        self.assertEqual(loaded_data.columns, self.columns)

        self.assertEqual(len(loaded_data.rows), 3)
        self.assertEqual(loaded_data.rows[0], self.row1)
        self.assertEqual(loaded_data.rows[1], self.row2)
        self.assertEqual(loaded_data.rows[2], self.row3)

    def test_settings(self):
        """Test that metadata settings can be extracted from a raw data
        file."""
        self._write_raw_data()

        name, sample_rate, columns = settings(self.path)
        self.assertEqual(name, self.daq_type)
        self.assertEqual(sample_rate, self.sample_rate)
        self.assertEqual(columns, self.columns)

    def test_raw_data_reader(self):
        """Test that data can be read from a file incrementally."""
        self._write_raw_data(include_rows=True)

        with RawDataReader(self.path) as reader:
            self.assertEqual(reader.daq_type, self.daq_type)
            self.assertEqual(reader.sample_rate, self.sample_rate)
            self.assertEqual(reader.columns, self.columns)

            # all columns are read as strings by default
            self.assertEqual(list(map(str, self.row1)), next(reader))
            self.assertEqual(list(map(str, self.row2)), next(reader))
            self.assertEqual(list(map(str, self.row3)), next(reader))

    def test_raw_data_reader_with_type_conversions(self):
        """Test that data can be read incrementally and that data can be
        converted to numeric types."""
        columns = ['timestamp', 'ch1', 'ch2', 'ch3', 'TRG']
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=columns)

        row1 = [1, 1.0, 2.0, 3.0, 'A']
        row2 = [2, 4.0, 5.0, 6.0, 'B']
        row3 = [3, 7.0, 8.0, 9.0, 'C']

        data.append(row1)
        data.append(row2)
        data.append(row3)
        write(data, self.path)

        with RawDataReader(self.path, convert_data=True) as reader:
            self.assertEqual(reader.daq_type, self.daq_type)
            self.assertEqual(reader.sample_rate, self.sample_rate)
            self.assertEqual(reader.columns, data.columns)

            self.assertEqual(row1, next(reader))
            self.assertEqual(row2, next(reader))
            self.assertEqual(row3, next(reader))

    def test_raw_data_writer(self):
        """Test that data can be written incrementally"""

        rows = [self.row1, self.row2, self.row3]

        self.assertFalse(self.path.exists())
        with RawDataWriter(self.path,
                           daq_type=self.daq_type,
                           sample_rate=self.sample_rate,
                           columns=self.columns) as writer:
            for row in rows:
                writer.writerow(row)
        self.assertTrue(self.path.exists())

    def test_raw_data_writer_writerows(self):
        """Test that data can be written in chunks."""

        rows = [self.row1, self.row2, self.row3]

        self.assertFalse(self.path.exists())
        with RawDataWriter(self.path,
                           daq_type=self.daq_type,
                           sample_rate=self.sample_rate,
                           columns=self.columns) as writer:
            writer.writerows(rows)
        self.assertTrue(self.path.exists())

        loaded_data = RawData.load(self.path)
        self.assertEqual(loaded_data.daq_type, self.daq_type)
        self.assertEqual(loaded_data.sample_rate, self.sample_rate)
        self.assertEqual(loaded_data.columns, self.columns)
        self.assertEqual(len(loaded_data.rows), 3)

    def test_raw_data_writer_initialization(self):
        """Test that writer can be manually opened and closed"""

        rows = [self.row1, self.row2, self.row3]

        self.assertFalse(self.path.exists())
        writer = RawDataWriter(self.path,
                               daq_type=self.daq_type,
                               sample_rate=self.sample_rate,
                               columns=self.columns)
        writer.__enter__()
        writer.writerows(rows)
        writer.__exit__()

        self.assertTrue(self.path.exists())
        loaded_data = RawData.load(self.path)

        self.assertEqual(loaded_data.daq_type, self.daq_type)
        self.assertEqual(loaded_data.sample_rate, self.sample_rate)
        self.assertEqual(loaded_data.columns, self.columns)

        self.assertEqual(len(loaded_data.rows), 3)

    def test_sample_data(self):
        """Test that sample data can be generated for testing purposes."""
        channels = ['ch1', 'ch2', 'ch3']
        data = sample_data(rows=5, ch_names=channels)
        self.assertEqual(5, len(data.rows))
        self.assertEqual(['timestamp', 'ch1', 'ch2', 'ch3', 'TRG'],
                         data.columns)
        self.assertEqual(1, data.rows[0][0])
        self.assertEqual(5, data.rows[4][0])

        # Test data range of channel columns
        for row in data.rows:
            for col in row[1:-1]:
                self.assertTrue(col >= -1000 and col <= 1000)

    def test_sample_data_with_triggers(self):
        """Test generating sample data with triggers at pre-defined timestamps.
        This functionality is used primarily for other unit tests."""
        channels = ['ch1', 'ch2', 'ch3']
        data = sample_data(rows=10,
                           ch_names=channels,
                           triggers=[(1, 'A'), (5, 'B'), (10, 'C')])
        self.assertEqual(10, len(data.rows))

        trg_col = data.columns.index('TRG')
        self.assertEqual('A', data.rows[0][trg_col])
        self.assertEqual('B', data.rows[4][trg_col])
        self.assertEqual('C', data.rows[9][trg_col])

    def test_raw_data_numeric_channels(self):
        """Tests that data channels can be extracted for analysis."""

        columns = ['timestamp', 'ch1', 'ch2', 'ch3', 'TRG']
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=columns)

        row1 = [1, 1.0, 2.0, 3.0, '0.0']
        row2 = [2, 4.0, 5.0, 6.0, 'A']
        row3 = [3, 7.0, 8.0, 9.0, '0.0']

        data.append(row1)
        data.append(row2)
        data.append(row3)

        self.assertEqual(['ch1', 'ch2', 'ch3'], data.channels)

        # Test numeric data frame
        dataframe = pd.DataFrame(data=[[1, 1.0, 2.0, 3.0], [2, 4.0, 5.0, 6.0],
                                       [3, 7.0, 8.0, 9.0]],
                                 columns=['timestamp', 'ch1', 'ch2', 'ch3'])
        self.assertTrue(np.all(dataframe == data.numeric_data))

    def test_data_by_channel(self):
        """Tests that data can be structured in column-order for analysis."""

        columns = ['timestamp', 'ch1', 'ch2', 'TRG']
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=columns)

        data.append([1, 1.0, 2.0, '0.0'])
        data.append([2, 4.0, 5.0, 'A'])
        data.append([3, 7.0, 8.0, '0.0'])

        arr, _ = data.by_channel()
        self.assertEqual((2, 3), arr.shape)

        self.assertTrue(len(data.channels), len(arr))
        self.assertTrue(np.all(arr[0] == [1.0, 4.0, 7.0]),
                        "Should have ch1 data")
        self.assertTrue(np.all(arr[1] == [2.0, 5.0, 8.0]),
                        "Should have ch2 data")

    def test_data_by_channel_applies_transformation(self):
        """Tests that data correctly structured, can have a transformation applied to it."""
        columns = ['timestamp', 'ch1', 'ch2', 'TRG']
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=columns)

        data.append([1, 1.0, 2.0, '0.0'])
        data.append([2, 4.0, 5.0, 'A'])
        data.append([3, 7.0, 8.0, '0.0'])

        transform = mock()
        # note data here should be returned as a nd.array. for mocking we don't care as much
        when(RawData).apply_transform(any(), transform).thenReturn((data, self.sample_rate))
        resp, fs = data.by_channel(transform=transform)

        self.assertEqual(self.sample_rate, fs)
        self.assertEqual(resp, data)
        verify(RawData, times=1).apply_transform(any(), transform)

    def test_data_by_channel_map(self):
        """Tests that when given a channel map, it will filter the numeric columns.

        We assume most trigger columns will be removed (due to being read in as a string or object),
        however some are numeric trigger channels and may require filtering with a channel map.
        Other cases could be dropping bad channels.
        """
        columns = ['timestamp', 'ch1', 'ch2', 'TRG']
        channel_map = [1, 1, 0]
        expected_channels = ['ch1', 'ch2']
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=columns)

        data.append([1, 1.0, 2.0, 0.0])
        data.append([2, 4.0, 5.0, 0.0])
        data.append([3, 7.0, 8.0, 0.0])

        arr, channels, _ = data.by_channel_map(channel_map=channel_map)

        self.assertEqual((2, 3), arr.shape)

        self.assertTrue(len(data.channels), len(arr))
        self.assertTrue(np.all(arr[0] == [1.0, 4.0, 7.0]),
                        "Should have ch1 data")
        self.assertTrue(np.all(arr[1] == [2.0, 5.0, 8.0]),
                        "Should have ch2 data")
        self.assertEqual(channels, expected_channels)

    def test_data_by_channel_map_applies_transformation(self):
        """Tests that when given a channel map, it will filter the numeric columns.

        We assume most trigger columns will be removed (due to being read in as a string or object),
        however some are numeric trigger channels and may require filtering with a channel map.
        Other cases could be dropping bad channels.
        """
        columns = ['timestamp', 'ch1', 'ch2', 'TRG']
        channel_map = [1, 1, 1]
        expected_channels = ['ch1', 'ch2', 'TRG']
        data = RawData(daq_type=self.daq_type,
                       sample_rate=self.sample_rate,
                       columns=columns)

        data.append([1, 1.0, 2.0, 0.0])
        data.append([2, 4.0, 5.0, 0.0])
        data.append([3, 7.0, 8.0, 0.0])

        transform = mock()
        expected_output, expected_fs = data.by_channel()
        # note data here should be returned as a nd.array. for mocking we don't care as much
        when(RawData).by_channel(transform).thenReturn((expected_output, expected_fs))
        _, channels, fs = data.by_channel_map(channel_map=channel_map, transform=transform)

        self.assertEqual(expected_fs, fs)
        self.assertEqual(expected_channels, channels)
        verify(RawData, times=1).by_channel(transform)


class Test1020(unittest.TestCase):
    """Tests for 10-20 channel mapping functions."""

    def test_get_1020_channels(self):
        """Tests that the 10-20 channel map is correctly generated."""
        channels = get_1020_channels()
        self.assertEqual(35, len(channels))
        self.assertTrue(isinstance(channels[0], str))

    def test_get_1020_channel_map(self):
        """Tests that the 10-20 channel map is correctly generated."""
        # all but the last channel are valid 10-20 channels
        channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'invalid']
        channel_map = get_1020_channel_map(channels)
        self.assertEqual(10, len(channel_map))
        self.assertEqual(0, channel_map[-1])
        for i in range(len(channels) - 1):
            self.assertEqual(1, channel_map[i])


if __name__ == '__main__':
    unittest.main()
