"""Functions for generating mock data to be used for testing/development."""

import logging

from past.builtins import range
from bcipy.signal.generator.generator import gen_random_data

log = logging.getLogger(__name__)


def advance_to_row(filehandle, rownum):
    """Utility function to advance a file cursor to the given row."""
    for _ in range(rownum - 1):
        filehandle.readline()

# pylint: disable=too-few-public-methods


class _DefaultEncoder():
    """Encodes data by returning the raw data."""

    # pylint: disable=no-self-use
    def encode(self, data):
        """Encode the data that will be output by the file_data generator."""
        return data


def random_data(encoder=_DefaultEncoder(), low=-1000, high=1000,
                channel_count=25):
    """Generates random EEG-like data encoded according to the provided encoder.

    Returns
    -------
        packet of data, which decodes into a list of floats in the range low
            to high with channel_count number of items.
    """

    while True:
        sensor_data = gen_random_data(low, high, channel_count)
        yield encoder.encode(sensor_data)


def file_data(filename, header_row=3, encoder=_DefaultEncoder()):
    """Generates data from a source file and encodes it according to the
    provided encoder.

    Parameters
    ----------
        filename: str
            Name of file containing a sample EEG session output. This file will
            be the source of the generated data. File should be a csv file.
        header_row: int, optional
            Row with the header data (channel names); the default is 3,
            assuming the first 2 rows are for metadata.
        encoder : Encoder, optional
            Used to encode the output data.

    """

    with open(filename, 'r') as infile:
        # advance to first data row, since channel names are unused.
        advance_to_row(infile, header_row + 1)

        while True:
            line = infile.readline()
            if not line:
                break
            sensor_data = list(map(data_value, line.split(",")))
            yield encoder.encode(sensor_data)


def data_value(value: str) -> float:
    """Convert to a float; some trigger values are strings, rather than
    numbers (ex. indicating the letter); convert these to 1.0."""
    if value:
        try:
            return float(value)
        except ValueError:
            return 1.0
    else:
        # empty string
        return 0.0
