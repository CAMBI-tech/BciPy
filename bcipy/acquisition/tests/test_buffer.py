# pylint: disable=no-self-use
"""Tests for the sqlite-backed data Buffer."""
import timeit

import unittest
import pytest
from bcipy.acquisition.buffer import Buffer
from bcipy.acquisition.record import Record
from bcipy.acquisition.util import mock_data

class _Timer():
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
    """Tests for the buffer module."""

    def test_buffer(self):
        """Test Buffer functionality."""

        n_records = 15000
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        buf = Buffer(channels=channels, chunksize=10000)

        append_timer = _Timer()
        timevalues = {}
        for i, data in enumerate(mock_data(n_records, channel_count)):
            timestamp = float(i)
            if i % 1000 == 0:
                timevalues[timestamp] = data
            with append_timer:
                buf.append(Record(data, timestamp, None))

        self.assertEqual(buf.start_time, 0.0)
        starttime = 0.0
        rows = buf.query(start=starttime, end=starttime +
                         1.0, field='timestamp')

        self.assertEqual(len(rows), 1, "Results should not include end value.")

        self.assertEqual(rows[0].timestamp, starttime)
        self.assertEqual(rows[0].data, timevalues[starttime])

        start_time = 1000.0
        end_time = 2000.0
        rows = buf.query(start=start_time, end=end_time, field='timestamp')
        self.assertEqual(len(rows), start_time)
        self.assertEqual(rows[0].data, timevalues[start_time])

        rows = buf.query(start=buf.start_time, field='timestamp')
        self.assertEqual(
            len(rows), n_records, "Providing only the start should return the rest.")
        buf.cleanup()

    def test_latest(self):
        """Test query for most recent items."""
        n_records = 1000
        latest_n = 100
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        buf = Buffer(channels=channels)

        latest = []
        for i, data in enumerate(mock_data(n_records, channel_count)):
            timestamp = float(i)
            if i >= n_records - latest_n:
                latest.append((data, timestamp, i+1))
            buf.append(Record(data, timestamp, None))

        rows = buf.latest(latest_n)
        for j, item in enumerate(reversed(latest)):
            self.assertEqual(item, rows[j])
        buf.cleanup()

    def test_len(self):
        """Test buffer len."""
        n_records = 1000
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        buf = Buffer(channels=channels)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            buf.append(Record(data, float(i), None))

        self.assertEqual(len(buf), n_records)
        buf.cleanup()

    def test_query_before_flush(self):
        """If a query is made before chunksize records have been written,
        the data should still be available."""

        n_records = 1000
        channel_count = 25
        channels = ["ch" + str(c) for c in range(channel_count)]

        buf = Buffer(channels=channels, chunksize=10000)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            timestamp = float(i)
            buf.append(Record(data, timestamp, None))

        rows = buf.query(start=buf.start_time, field='timestamp')
        self.assertEqual(len(rows), n_records)
        self.assertEqual(len(buf.all()), n_records)

        buf.cleanup()

    def test_query_data(self):
        """Test querying for data."""
        n_records = 20

        channels = ["ch1", "ch2", "TRG"]
        trg_index = -1
        channel_count = len(channels)

        buf = Buffer(channels=channels)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            data[trg_index] = 1.0 if i >= 10 else 0.0
            timestamp = float(i)
            buf.append(Record(data, timestamp, None))

        rows = buf.query_data(filters=[("TRG", ">", 0)],
                              ordering=("timestamp", "asc"),
                              max_results=1)

        self.assertEqual(len(rows), 1, "Should have limited to max_results.")
        self.assertEqual(rows[0].data[trg_index], 1.0,
                         "Should get filtered data.")
        self.assertEqual(rows[0].timestamp, 10.0,
                         "Should get first instance")
        buf.cleanup()

    def test_query_with_invalid_filter_field(self):
        """Test query with invalid filter field."""
        n_records = 20

        channels = ["ch1", "ch2", "TRG"]
        channel_count = len(channels)

        buf = Buffer(channels=channels)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            timestamp = float(i)
            buf.append(Record(data, timestamp, None))

        with pytest.raises(Exception):
            buf.query_data(filters=[("ch3", ">", 0)])
        buf.cleanup()

    def test_query_with_invalid_filter_op(self):
        """Test query with invalid filter operator."""
        n_records = 20

        channels = ["ch1", "ch2", "TRG"]
        channel_count = len(channels)

        buf = Buffer(channels=channels)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            timestamp = float(i)
            buf.append(Record(data, timestamp, None))

        with pytest.raises(Exception):
            buf.query_data(filters=[("TRG", "> 0; DROP TABLE data; --", 0)])

        buf.cleanup()

    def test_query_with_invalid_order_field(self):
        """Test query with invalid order field."""
        n_records = 20

        channels = ["ch1", "ch2", "TRG"]
        trg_index = -1
        channel_count = len(channels)

        buf = Buffer(channels=channels)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            data[trg_index] = 1.0 if i >= 10 else 0.0
            timestamp = float(i)
            buf.append(Record(data, timestamp, None))

        with pytest.raises(Exception):
            buf.query_data(ordering=("ch3", "asc"))
        buf.cleanup()

    def test_query_with_invalid_order_direction(self):
        """Test query with invalid order direction"""
        n_records = 20

        channels = ["ch1", "ch2", "TRG"]
        trg_index = -1
        channel_count = len(channels)

        buf = Buffer(channels=channels)

        for i, data in enumerate(mock_data(n_records, channel_count)):
            data[trg_index] = 1.0 if i >= 10 else 0.0
            timestamp = float(i)
            buf.append(Record(data, timestamp, None))

        with pytest.raises(Exception):
            buf.query_data(ordering=("ch1", "ascending"))
        buf.cleanup()
