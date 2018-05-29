
import timeit

import numpy as np
from bcipy.acquisition.buffer import Buffer
from bcipy.acquisition.record import Record
import pytest
import unittest


def _mockdata(n, channel_count):
    """Generates for mock data"""
    i = 0
    while i < n:
        yield [np.random.uniform(-1000, 1000) for cc in range(channel_count)]
        i += 1


class _Timer(object):
    """Repeatedly use _Timer to accumulate timing information. """

    def __init__(self):
        super(_Timer, self).__init__()
        self.timings = []
        self._start = None

    def __enter__(self):
        self._start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stop = timeit.default_timer()
        self.timings.append(stop - self._start)
        self._start = None


class TestBuffer(unittest.TestCase):
    def test_buffer(self):
        """Test Buffer functionality."""

        n = 15000
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        b = Buffer(channels=channels, chunksize=10000)

        append_timer = _Timer()
        timevalues = {}
        for i, d in enumerate(_mockdata(n, channel_count)):
            timestamp = float(i)
            if i % 1000 == 0:
                timevalues[timestamp] = d
            with append_timer:
                b.append(Record(d, timestamp))

        self.assertEqual(b.start_time, 0.0)
        starttime = 0.0
        rows = b.query(start=starttime, end=starttime + 1.0)

        self.assertEqual(len(rows), 1, "Results should not include end value.")

        self.assertEqual(rows[0].timestamp, starttime)
        self.assertEqual(rows[0].data, timevalues[starttime])

        st = 1000.0
        et = 2000.0
        rows = b.query(start=st, end=et)
        self.assertEqual(len(rows), st)
        self.assertEqual(rows[0].data, timevalues[st])

        rows = b.query(start=b.start_time)
        self.assertEqual(
            len(rows), n, "Providing only the start should return the rest.")
        b.cleanup()

    def test_latest(self):
        """Test query for most recent items."""
        n = 1000
        latest_n = 100
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        b = Buffer(channels=channels)

        latest = []
        for i, d in enumerate(_mockdata(n, channel_count)):
            timestamp = float(i)
            if i >= n - latest_n:
                latest.append((d, timestamp))
            b.append(Record(d, timestamp))

        rows = b.latest(latest_n)
        for j, item in enumerate(reversed(latest)):
            self.assertEqual(item, rows[j])
        b.cleanup()

    def test_len(self):
        """Test buffer len."""
        n = 1000
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        b = Buffer(channels=channels)

        for i, d in enumerate(_mockdata(n, channel_count)):
            b.append(Record(d, float(i)))

        self.assertEqual(len(b), n)
        b.cleanup()

    def test_query_before_flush(self):
        """If a query is made before chunksize records have been written,
        the data should still be available."""

        n = 1000
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        b = Buffer(channels=channels, chunksize=10000)

        for i, d in enumerate(_mockdata(n, channel_count)):
            timestamp = float(i)
            b.append(Record(d, timestamp))

        rows = b.query(start=b.start_time)
        self.assertEqual(len(rows), n)
        self.assertEqual(len(b.all()), n)

        b.cleanup()

    def test_query_data(self):
        n = 20

        channels = ["ch1", "ch2", "TRG"]
        trg_index = -1
        channel_count = len(channels)

        b = Buffer(channels=channels)

        for i, d in enumerate(_mockdata(n, channel_count)):
            d[trg_index] = 1.0 if i >= 10 else 0.0
            timestamp = float(i)
            b.append(Record(d, timestamp))

        rows = b.query_data(filters=[("TRG", ">", 0)],
                            ordering=("timestamp", "asc"),
                            max_results=1)

        self.assertEqual(len(rows), 1, "Should have limited to max_results.")
        self.assertEqual(rows[0].data[trg_index], 1.0,
                         "Should get filtered data.")
        self.assertEqual(rows[0].timestamp, 10.0,
                         "Should get first instance")
        b.cleanup()

    def test_query_with_invalid_filter_field(self):
        n = 20

        channels = ["ch1", "ch2", "TRG"]
        channel_count = len(channels)

        b = Buffer(channels=channels)

        for i, d in enumerate(_mockdata(n, channel_count)):
            timestamp = float(i)
            b.append(Record(d, timestamp))

        with pytest.raises(Exception):
            b.query_data(filters=[("ch3", ">", 0)])
        b.cleanup()

    def test_query_with_invalid_filter_op(self):
        n = 20

        channels = ["ch1", "ch2", "TRG"]
        channel_count = len(channels)

        b = Buffer(channels=channels)

        for i, d in enumerate(_mockdata(n, channel_count)):
            timestamp = float(i)
            b.append(Record(d, timestamp))

        with pytest.raises(Exception):
            b.query_data(filters=[("TRG", "> 0; DROP TABLE data; --", 0)])

        b.cleanup()

    def test_query_with_invalid_order_field(self):
        n = 20

        channels = ["ch1", "ch2", "TRG"]
        trg_index = -1
        channel_count = len(channels)

        b = Buffer(channels=channels)

        for i, d in enumerate(_mockdata(n, channel_count)):
            d[trg_index] = 1.0 if i >= 10 else 0.0
            timestamp = float(i)
            b.append(Record(d, timestamp))

        with pytest.raises(Exception):
            b.query_data(ordering=("ch3", "asc"))
        b.cleanup()

    def test_query_with_invalid_order_direction(self):
        n = 20

        channels = ["ch1", "ch2", "TRG"]
        trg_index = -1
        channel_count = len(channels)

        b = Buffer(channels=channels)

        for i, d in enumerate(_mockdata(n, channel_count)):
            d[trg_index] = 1.0 if i >= 10 else 0.0
            timestamp = float(i)
            b.append(Record(d, timestamp))

        with pytest.raises(Exception):
            b.query_data(ordering=("ch1", "ascending"))
        b.cleanup()
