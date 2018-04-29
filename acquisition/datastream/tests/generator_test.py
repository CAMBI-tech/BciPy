from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
from builtins import next
from datastream import generator
from mock import mock_open, patch
from past.builtins import map, range
import unittest


class CustomEncoder(object):
    """Encodes data by prefixing with the count."""

    def __init__(self):
        super(CustomEncoder, self).__init__()
        self.counter = 0

    def encode(self, data):
        self.counter += 1
        return (self.counter, data)


class TestGenerator(unittest.TestCase):
    """Tests for Generator"""

    def test_random_generator(self):
        """Test default parameters for random generator"""
        data = []
        gen = generator.random_data()

        for i in range(100):
            data.append(next(gen))

        self.assertEqual(len(data), 100)

    def test_random_high_low_values(self):
        """Random generator should allow user to set value ranges."""
        channel_count = 10
        low = -100
        high = 100
        gen = generator.random_data(low=-100, high=100,
                                    channel_count=channel_count)
        data = []
        for i in range(100):
            data.append(next(gen))

        self.assertEqual(len(data), 100)

        for record in data:
            self.assertEqual(len(record), channel_count)
            for value in record:
                self.assertTrue(value >= low and value <= high)

    def test_random_with_custom_encoder(self):
        """Random generator should allow a custom encoder."""

        data = []
        channel_count = 10
        gen = generator.random_data(encoder=CustomEncoder(),
                                    channel_count=channel_count)

        for i in range(100):
            data.append(next(gen))

        self.assertEqual(len(data), 100)
        for count, record in data:
            self.assertEqual(len(record), channel_count)

        self.assertEqual(data[0][0], 1)
        self.assertEqual(data[99][0], 100)

    def test_file_generator(self):
        """Should stream data from a file."""
        col_count = 3
        row_count = 100

        header = ['col1,col2,col3']
        file_data = [[float(cnum + rnum) for cnum in range(col_count)]
                     for rnum in range(row_count)]
        rows = map(lambda x: ','.join(map(str, x)), file_data)
        test_data = '\n'.join(header + rows)

        with patch('datastream.generator.open',
                   mock_open(read_data=test_data), create=True):

            data = []
            gen = generator.file_data(filename='foo', header_row=1)
            for i in range(row_count):
                data.append(next(gen))

            self.assertEqual(len(data), row_count)
            for i, row in enumerate(data):
                self.assertEqual(row, file_data[i])

    def test_file_generator_end(self):
        """Should throw an exception when all data has been consumed"""
        col_count = 3
        row_count = 10

        header = ['col1,col2,col3']
        file_data = [[float(cnum + rnum) for cnum in range(col_count)]
                     for rnum in range(row_count)]
        rows = map(lambda x: ','.join(map(str, x)), file_data)
        test_data = '\n'.join(header + rows)

        with patch('datastream.generator.open',
                   mock_open(read_data=test_data), create=True):

            data = []
            gen = generator.file_data(filename='foo', header_row=1)
            for i in range(row_count):
                data.append(next(gen))

            with pytest.raises(StopIteration):
                data.append(next(gen))

    def test_file_with_custom_encoder(self):
        """Should allow a custom encoder"""

        col_count = 3
        row_count = 100

        header = ['col1,col2,col3']
        file_data = [[float(cnum + rnum) for cnum in range(col_count)]
                     for rnum in range(row_count)]
        rows = map(lambda x: ','.join(map(str, x)), file_data)
        test_data = '\n'.join(header + rows)

        with patch('datastream.generator.open',
                   mock_open(read_data=test_data), create=True):

            data = []
            gen = generator.file_data(
                filename='foo', header_row=1, encoder=CustomEncoder())
            for i in range(row_count):
                data.append(next(gen))

            for count, record in data:
                self.assertEqual(len(record), col_count)

            self.assertEqual(data[0][0], 1)
            self.assertEqual(data[99][0], 100)
