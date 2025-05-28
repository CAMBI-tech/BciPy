import itertools as it
from queue import Empty, Queue


class DataSource:
    """Abstract base class for an object with data. Provided to the EEGFrame
    for streaming data."""

    def next(self):
        """Provide the next record."""
        raise NotImplementedError('Subclass must define the next method')

    def next_n(self, n: int, fast_forward=False):
        """Provides the next n records as a list"""
        raise NotImplementedError('Subclass must define the next_n method')


class QueueDataSource(DataSource):
    """Queue backed data source."""

    def __init__(self, q: Queue):
        self.queue = q
        self.wait = 0.1

    def next(self):
        """Provide the next record."""
        try:
            return self.queue.get()
        except Empty:
            raise StopIteration

    def next_n(self, n: int, fast_forward=False):
        """Provides the next n records as a list"""
        data = []
        while len(data) < n:
            try:
                record = self.queue.get(True, self.wait)
                data.append(record)
            except Empty:
                raise StopIteration
        return data


class GeneratorDataSource(DataSource):
    """DataSource that uses a provided python generator."""

    def __init__(self, gen):
        self.gen = gen

    def next(self):
        """Provide the next record."""
        return next(self.gen)

    def next_n(self, n, fast_forward=False):
        """Provides the next n records as a list"""
        return list(it.islice(self.gen, n))
