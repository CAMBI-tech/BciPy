# pylint: disable=too-few-public-methods,no-self-use
"""Tests for datastream generator module"""
from builtins import next
import unittest
import pytest
from past.builtins import map, range
from mock import mock_open, patch
from bcipy.acquisition.datastream.generator import random_data, file_data
from bcipy.acquisition.util import mock_data


class CustomEncoder():
    """Encodes data by prefixing with the count."""

    def __init__(self):
        super(CustomEncoder, self).__init__()
        self.counter = 0

    def encode(self, data):
        """Encode the data."""
        self.counter += 1
        return (self.counter, data)


class TestGenerator(unittest.TestCase):
    """Tests for Generator"""

    def test_random_generator(self):
        """Test default parameters for random generator"""
        gen = random_data()
        data = [next(gen) for _ in range(100)]
        self.assertEqual(len(data), 100)

    def test_random_high_low_values(self):
        """Random generator should allow user to set value ranges."""
        channel_count = 10
        low = -100
        high = 100
        gen = random_data(low=-100, high=100,
                          channel_count=channel_count)
        data = [next(gen) for _ in range(100)]

        self.assertEqual(len(data), 100)

        for record in data:
            self.assertEqual(len(record), channel_count)
            for value in record:
                self.assertTrue(low <= value <= high)

    def test_random_with_custom_encoder(self):
        """Random generator should allow a custom encoder."""

        channel_count = 10
        gen = random_data(encoder=CustomEncoder(),
                          channel_count=channel_count)

        data = [next(gen) for _ in range(100)]

        self.assertEqual(len(data), 100)
        for _count, record in data:
            self.assertEqual(len(record), channel_count)

        self.assertEqual(data[0][0], 1)
        self.assertEqual(data[99][0], 100)

    def test_file_generator(self):
        """Should stream data from a file."""
        row_count = 100
        header = ['col1,col2,col3']
        data = list(mock_data(row_count, len(header)))
        rows = map(lambda x: ','.join(map(str, x)), data)
        test_data = '\n'.join(header + rows)

        with patch('bcipy.acquisition.datastream.generator.open',
                   mock_open(read_data=test_data), create=True):

            gen = file_data(filename='foo', header_row=1)
            generated_data = [next(gen) for _ in range(row_count)]

            for i, row in enumerate(generated_data):
                self.assertEqual(row, data[i])

    def test_file_generator_end(self):
        """Should throw an exception when all data has been consumed"""
        row_count = 10

        header = ['col1,col2,col3']
        data = list(mock_data(row_count, len(header)))
        rows = map(lambda x: ','.join(map(str, x)), data)
        test_data = '\n'.join(header + rows)

        with patch('bcipy.acquisition.datastream.generator.open',
                   mock_open(read_data=test_data), create=True):
            gen = file_data(filename='foo', header_row=1)
            # exhaust the generator
            _generated_data = [next(gen) for _ in range(row_count)]

            with pytest.raises(StopIteration):
                data.append(next(gen))

    def test_file_with_custom_encoder(self):
        """Should allow a custom encoder"""

        col_count = 3
        row_count = 100

        header = ['col1,col2,col3']
        data = [[float(cnum + rnum) for cnum in range(col_count)]
                for rnum in range(row_count)]
        rows = map(lambda x: ','.join(map(str, x)), data)
        test_data = '\n'.join(header + rows)

        with patch('bcipy.acquisition.datastream.generator.open',
                   mock_open(read_data=test_data), create=True):

            gen = file_data(
                filename='foo', header_row=1, encoder=CustomEncoder())
            generated_data = [next(gen) for _ in range(row_count)]

            for _count, record in generated_data:
                self.assertEqual(len(record), col_count)

            self.assertEqual(generated_data[0][0], 1)
            self.assertEqual(generated_data[99][0], 100)
