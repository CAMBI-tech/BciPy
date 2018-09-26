"""Tests for the buffer_server module"""
import unittest
# pylint: disable=useless-import-alias
import bcipy.acquisition.buffer_server as buffer_server
from bcipy.acquisition.record import Record
from bcipy.acquisition.util import mock_data


class TestBufferServer(unittest.TestCase):
    """Tests for the buffer_server module."""

    def __init__(self, *args, **kwargs):
        super(TestBufferServer, self).__init__(*args, **kwargs)
        self.i = 1

    def _next_buf_name(self):
        name = 'buffer_{}.db'.format(self.i)
        self.i += 1
        return name

    def setUp(self):
        """Run before each test."""
        self.channel_count = 25
        self.channels = ["ch" + str(c) for c in range(self.channel_count)]
        self.pid = buffer_server.start(self.channels, self._next_buf_name())

    def tearDown(self):
        """Run after each test."""
        buffer_server.stop(self.pid)
        self.pid = None

    def test_count(self):
        """Test that the count of records is correct."""
        n_records = 500
        for i, data in enumerate(mock_data(n_records, self.channel_count)):
            buffer_server.append(self.pid, Record(data=data, timestamp=i,
                                                  rownum=None))

        self.assertEqual(buffer_server.count(self.pid), n_records)

    def test_get_data_slice(self):
        """Test querying for a slice of data."""

        data = list(mock_data(n_records=150, n_cols=self.channel_count))
        for i, record in enumerate(data):
            buffer_server.append(self.pid, Record(data=record, timestamp=i,
                                                  rownum=None))

        start = 10
        end = 20

        result = buffer_server.get_data(
            self.pid, start, end, field='timestamp')
        self.assertEqual([r.data for r in result], data[start:end], "Should \
            return the slice of data requested.")

    def test_query_data(self):
        """Test query_data method"""

        data = list(mock_data(n_records=150, n_cols=self.channel_count))
        last_channel = self.channels[-1]

        for record_index, record in enumerate(data):
            record[-1] = 1.0 if record_index >= 100 else 0.0
            buffer_server.append(self.pid, Record(data=record,
                                                  timestamp=record_index,
                                                  rownum=None))

        result = buffer_server.query(self.pid,
                                     filters=[(last_channel, ">", 0)],
                                     ordering=("timestamp", "asc"),
                                     max_results=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].data[-1], 1.0)
        self.assertEqual(result[0].timestamp, 100.0)

    def test_get_all_data(self):
        """Test method to get all data from buffer."""

        data = list(mock_data(n_records=150, n_cols=self.channel_count))
        for record_index, record in enumerate(data):
            buffer_server.append(self.pid, Record(data=record,
                                                  timestamp=record_index,
                                                  rownum=None))

        result = buffer_server.get_data(self.pid)
        self.assertEqual([r.data for r in result], data, "Should return all \
            data")

    def test_multiple_servers(self):
        """Test multiple concurrent servers."""
        pid2 = buffer_server.start(self.channels, self._next_buf_name())

        n_records = 200
        for count, data in enumerate(mock_data(n_records, self.channel_count)):
            if count % 2 == 0:
                buffer_server.append(self.pid, Record(data, count, None))
            else:
                buffer_server.append(pid2, Record(data, count, None))

        self.assertEqual(buffer_server.count(self.pid), n_records / 2)
        self.assertEqual(buffer_server.count(pid2), n_records / 2)

        server1_data = buffer_server.get_data(self.pid, 0, 5)
        server2_data = buffer_server.get_data(pid2, 0, 5)

        self.assertNotEqual(server1_data, server2_data)
        buffer_server.stop(pid2)
