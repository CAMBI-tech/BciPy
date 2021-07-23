"""Tests for BciPy raw data format."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bcipy.helpers.raw_data import (RawData, RawDataReader, RawDataWriter,
                                    load, sample_data, settings, write)


class TestRawData(unittest.TestCase):
    """Tests for raw_data format"""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        self.data_dir = f"{os.path.dirname(__file__)}/resources/"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test that a RawData structure can be initialized"""
        columns = ['timestamp', 'ch1', 'ch2', 'ch3', 'TRG']
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=columns)
        self.assertEqual(data.daq_type, 'DSI-24')
        self.assertEqual(data.sample_rate, 300.0)
        self.assertEqual(data.columns, columns)

    def test_write_with_no_data(self):
        """Test that raw data can be persisted to disk."""
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24',
                       sample_rate=300.0,
                       columns=['timestamp', 'ch1', 'ch2', 'ch3', 'TRG'])

        self.assertFalse(path.exists())
        write(data, path)
        self.assertTrue(path.exists())

    def test_load_with_no_data(self):
        """Test that the raw data format can be read by the module."""
        columns = ['timestamp', 'ch1', 'ch2', 'ch3', 'TRG']
        path = Path(self.temp_dir, 'test_raw_data.csv')
        existing_data = RawData(daq_type='DSI-24',
                                sample_rate=300.0,
                                columns=columns)
        write(existing_data, path)

        self.assertTrue(path.exists())
        data = load(path)

        self.assertEqual(data.daq_type, 'DSI-24')
        self.assertEqual(data.sample_rate, 300.0)
        self.assertEqual(data.columns, columns)

    def test_load_with_data(self):
        """Test that data can be loaded from a file."""
        columns = columns = ['timestamp', 'ch1', 'ch2', 'ch3']
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=columns)

        row1 = [1, 1.0, 2.0, 3.0]
        row2 = [2, 4.0, 5.0, 6.0]
        row3 = [3, 7.0, 8.0, 9.0]

        data.append(row1)
        data.append(row2)
        data.append(row3)

        self.assertFalse(path.exists())
        write(data, path)
        self.assertTrue(path.exists())

        loaded_data = load(path)

        self.assertEqual(loaded_data.daq_type, 'DSI-24')
        self.assertEqual(loaded_data.sample_rate, 300.0)
        self.assertEqual(loaded_data.columns, columns)

        self.assertEqual(len(loaded_data.rows), 3)
        self.assertEqual(loaded_data.rows[0], row1)
        self.assertEqual(loaded_data.rows[1], row2)
        self.assertEqual(loaded_data.rows[2], row3)

    def test_deserialization(self):
        """Test that the load function can be accessed through a class
        constructor."""
        columns = columns = ['timestamp', 'ch1', 'ch2', 'ch3']
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=columns)

        row1 = [1, 1.0, 2.0, 3.0]
        row2 = [2, 4.0, 5.0, 6.0]
        row3 = [3, 7.0, 8.0, 9.0]

        data.append(row1)
        data.append(row2)
        data.append(row3)

        write(data, path)
        self.assertTrue(path.exists())

        loaded_data = RawData.load(path)

        self.assertEqual(loaded_data.daq_type, 'DSI-24')
        self.assertEqual(loaded_data.sample_rate, 300.0)
        self.assertEqual(loaded_data.columns, columns)

        self.assertEqual(len(loaded_data.rows), 3)
        self.assertEqual(loaded_data.rows[0], row1)
        self.assertEqual(loaded_data.rows[1], row2)
        self.assertEqual(loaded_data.rows[2], row3)

    def test_settings(self):
        """Test that metadata settings can be extracted from a raw data
        file."""
        path = Path(self.temp_dir, 'test_raw_data.csv')
        write(
            RawData(daq_type='LSL',
                    sample_rate=256.0,
                    columns=['ch_a', 'ch_b', 'ch_c']), path)
        name, sample_rate, columns = settings(path)
        self.assertEqual(name, 'LSL')
        self.assertEqual(sample_rate, 256.0)
        self.assertEqual(columns, ['ch_a', 'ch_b', 'ch_c'])

    def test_raw_data_reader(self):
        """Test that data can be read from a file incrementally."""
        channels = ['ch1', 'ch2', 'ch3']
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=channels)

        row1 = [1.0, 2.0, 3.0]
        row2 = [4.0, 5.0, 6.0]
        row3 = [7.0, 8.0, 9.0]

        data.append(row1)
        data.append(row2)
        data.append(row3)

        self.assertFalse(path.exists())
        write(data, path)
        self.assertTrue(path.exists())

        with RawDataReader(path) as reader:
            self.assertEqual(reader.daq_type, 'DSI-24')
            self.assertEqual(reader.sample_rate, 300.0)
            self.assertEqual(reader.columns, data.columns)

            # all columns are read as strings by default
            self.assertEqual(list(map(str, row1)), next(reader))
            self.assertEqual(list(map(str, row2)), next(reader))
            self.assertEqual(list(map(str, row3)), next(reader))

    def test_raw_data_reader_with_type_conversions(self):
        """Test that data can be read incrementally and that data can be
        converted to numeric types."""
        columns = ['timestamp', 'ch1', 'ch2', 'ch3', 'TRG']
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=columns)

        row1 = [1, 1.0, 2.0, 3.0, 'A']
        row2 = [2, 4.0, 5.0, 6.0, 'B']
        row3 = [3, 7.0, 8.0, 9.0, 'C']

        data.append(row1)
        data.append(row2)
        data.append(row3)

        self.assertFalse(path.exists())
        write(data, path)
        self.assertTrue(path.exists())

        with RawDataReader(path, convert_data=True) as reader:
            self.assertEqual(reader.daq_type, 'DSI-24')
            self.assertEqual(reader.sample_rate, 300.0)
            self.assertEqual(reader.columns, data.columns)

            self.assertEqual(row1, next(reader))
            self.assertEqual(row2, next(reader))
            self.assertEqual(row3, next(reader))

    def test_raw_data_writer(self):
        """Test that data can be written incrementally"""
        columns = ['timestamp', 'ch1', 'ch2', 'ch3']
        path = Path(self.temp_dir, 'test_raw_data_writer.csv')
        rows = [[1, 1.0, 2.0, 3.0], [2, 4.0, 5.0, 6.0], [3, 7.0, 8.0, 9.0]]

        self.assertFalse(path.exists())
        with RawDataWriter(path,
                           daq_type='DSI-24',
                           sample_rate=300.0,
                           columns=columns) as writer:
            for row in rows:
                writer.writerow(row)
        self.assertTrue(path.exists())

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

        columns = columns = ['timestamp', 'ch1', 'ch2', 'ch3', 'TRG']
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=columns)

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
        path = Path(self.temp_dir, 'test_raw_data.csv')
        data = RawData(daq_type='DSI-24', sample_rate=300.0, columns=columns)

        data.append([1, 1.0, 2.0, '0.0'])
        data.append([2, 4.0, 5.0, 'A'])
        data.append([3, 7.0, 8.0, '0.0'])

        arr = data.by_channel()
        self.assertEqual((2, 3), arr.shape)

        self.assertTrue(len(data.channels), len(arr))
        self.assertTrue(np.all(arr[0] == [1.0, 4.0, 7.0]), "Should have ch1 data")
        self.assertTrue(np.all(arr[1] == [2.0, 5.0, 8.0]), "Should have ch2 data")
