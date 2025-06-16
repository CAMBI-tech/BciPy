"""Functions for generating mock data to be used for testing/development."""

from typing import Callable, Generator, List, Optional, TextIO, Any

from past.builtins import range

from bcipy.config import DEFAULT_ENCODING
from bcipy.signal.generator.generator import gen_random_data


def advance_to_row(filehandle: TextIO, rownum: int) -> None:
    """Utility function to advance a file cursor to the given row.

    Args:
        filehandle (TextIO): The file handle to advance.
        rownum (int): The target row number to advance to (1-indexed).
    """
    for _ in range(rownum - 1):
        filehandle.readline()


# pylint: disable=too-few-public-methods


class _DefaultEncoder:
    """Encodes data by returning the raw data."""

    # pylint: disable=no-self-use
    def encode(self, data: Any) -> Any:
        """Encodes the input data.

        Args:
            data (Any): The data to be encoded.

        Returns:
            Any: The raw input data, as no actual encoding is performed.
        """
        return data


def generator_with_args(generator_fn: Callable[..., Generator], **generator_args: Any) -> Callable[..., Generator]:
    """Constructs a generator with the given arguments.

    Args:
        generator_fn (Callable[..., Generator]): A generator function.
        **generator_args (Any): Keyword arguments to be passed to the generator_fn.

    Returns:
        Callable[..., Generator]: A function that creates a generator using the given args.
    """

    def factory(**args: Any) -> Generator:
        return generator_fn(**{**generator_args, **args})

    return factory


def random_data_generator(encoder: _DefaultEncoder = _DefaultEncoder(),
                          low: int = -1000,
                          high: int = 1000,
                          channel_count: int = 25) -> Generator[Any, None, None]:
    """Generator that outputs random EEG-like data encoded according to the provided encoder.

    Args:
        encoder (_DefaultEncoder, optional): An encoder object to encode the output data.
                                             Defaults to _DefaultEncoder().
        low (int, optional): The lower bound for random data generation. Defaults to -1000.
        high (int, optional): The upper bound for random data generation. Defaults to 1000.
        channel_count (int, optional): The number of channels (items) in the generated data.
                                       Defaults to 25.

    Yields:
        Any: A packet of data, which decodes into a list of floats in the range
             low to high with `channel_count` number of items, encoded by the encoder.
    """

    while True:
        sensor_data = gen_random_data(low, high, channel_count)
        yield encoder.encode(sensor_data)


def file_data_generator(filename: str,
                        header_row: int = 3,
                        encoder: _DefaultEncoder = _DefaultEncoder(),
                        channel_count: Optional[int] = None) -> Generator[List[float], None, None]:
    """Generates data from a source file and encodes it according to the
    provided encoder.

    Args:
        filename (str): Name of file containing a sample EEG session output.
                        This file will be the source of the generated data.
                        File should be a csv file.
        header_row (int, optional): Row with the header data (channel names).
                                    The default is 3, assuming the first 2 rows
                                    are for metadata.
        encoder (_DefaultEncoder, optional): Used to encode the output data.
                                            Defaults to _DefaultEncoder().
        channel_count (Optional[int], optional): If provided this is used to truncate
                                                  the data to the given number of channels.
                                                  Defaults to None.

    Yields:
        List[float]: A list of float values representing sensor data from the file,
                     encoded by the encoder.
    """

    with open(filename, 'r', encoding=DEFAULT_ENCODING) as infile:
        # advance to first data row, since channel names are unused.
        advance_to_row(infile, header_row + 1)

        while True:
            line = infile.readline()
            if not line:
                break
            sensor_data = list(map(data_value, line.split(",")))
            if channel_count:
                sensor_data = sensor_data[0:channel_count]
            yield encoder.encode(sensor_data)


def data_value(value: str) -> float:
    """Converts a string value to a float.

    Some trigger values might be strings (e.g., indicating a letter),
    which are converted to 1.0. Empty strings are converted to 0.0.

    Args:
        value (str): The string value to convert.

    Returns:
        float: The converted float value.
    """
    if value:
        try:
            return float(value)
        except ValueError:
            return 1.0
    else:
        # empty string
        return 0.0
