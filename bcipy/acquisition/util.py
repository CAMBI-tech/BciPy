"""Defines utility classes and functions used by the acquisition module."""
import multiprocessing
import threading
import numpy as np


def mock_data(n_records: int, n_cols: int, low: int = -1000, high: int = 1000):
    """Generator for mock data that streams a finite number of
    records.

    Parameters:
    -----------
        n_records: number of records to generate
        n_cols: number of columns of each record
        low: optional; min value for each item; default is -1000
        high: optional; max value for each item; default is 1000

    Returns:
    --------
        A generator that yields arrays of random data
    """
    return (mock_record(n_cols, low, high) for _i in range(n_records))

# def mock_data_stream(n_records: int, length: int, low: int= -1000,
#     high: int = 1000):
#     """Generator for mock data that streams continuously.

#     Parameters:
#     -----------
#         n_records: number of records to generate
#         length: length of each record
#         low: optional; min value for each item; default is -1000
#         high: optional; max value for each item; default is 1000

#     Returns:
#     --------
#         A generator that yields arrays of random data
#     """
#     while True:
#         yield mock_record(length, low, high)


def mock_record(n_cols: int = 25, low: int = -1000, high: int = 1000):
    """Create an list of random data.

    Parameters:
    -----------
        n_cols - number of columns in the generated record.
        low - minimum value in each position
        high - maximum value in each position
    Returns:
    --------
        list of random values
    """
    return [np.random.uniform(low, high) for _cc in range(n_cols)]


class StoppableProcess(multiprocessing.Process):
    """Process class with a stop() method. The process itself has to check
    regularly for the running() condition.
    """

    def __init__(self, *args, **kwargs):
        super(StoppableProcess, self).__init__(*args, **kwargs)
        self._stopper = multiprocessing.Event()

    def stop(self):
        """Stop the process."""
        self._stopper.set()

    def running(self):
        """Test if the process is currently running."""
        return not self._stopper.is_set()

    def stopped(self):
        """Test if the process is stopped."""
        return self._stopper.is_set()


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the running() condition.
    """

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stopper = threading.Event()

    def stop(self):
        """Stop the thread."""
        self._stopper.set()

    def running(self):
        """Test if the thread is currently running."""
        return not self._stopper.is_set()

    def stopped(self):
        """Test if the thread is currently stopped."""
        return self._stopper.is_set()
