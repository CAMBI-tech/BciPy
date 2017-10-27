from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import timeit

import numpy as np
from buffer import Buffer
from client import _StoppableThread
from record import Record


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
        self.times = []
        self._start = None

    def __enter__(self):
        self._start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stop = timeit.default_timer()
        self.times.append(stop - self._start)
        self._start = None


def test_buffer():
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

    # Performance assertion; TODO: is this a reasonable requirement? Does it
    # make sense to test max insert time?
    assert sum(append_timer.times) / len(append_timer.times) < 1 / 600

    assert b.start_time == 0.0
    starttime = 0.0
    rows = b.query(start=starttime, end=starttime + 1.0)

    assert len(rows) == 1, "Results should not include the end value."

    assert rows[0].timestamp == starttime
    assert rows[0].data == timevalues[starttime]

    st = 1000.0
    et = 2000.0
    rows = b.query(start=st, end=et)
    assert len(rows) == st
    assert rows[0].data == timevalues[st]

    rows = b.query(start=b.start_time)
    assert len(rows) == n, "Providing only the start should return the rest."


def test_latest():
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
        assert item == rows[j]


def test_len():
    """Test buffer len."""
    n = 1000
    channel_count = 25
    channels = ["ch" + str(c) for c in range(channel_count)]

    b = Buffer(channels=channels)

    for i, d in enumerate(_mockdata(n, channel_count)):
        b.append(Record(d, float(i)))

    assert len(b) == n


def test_query_before_flush():
    """If a query is made before chunksize records have been written, the data
    should still be available."""

    n = 1000
    channel_count = 25
    channels = ["ch" + str(c) for c in range(channel_count)]

    b = Buffer(channels=channels, chunksize=10000)

    for i, d in enumerate(_mockdata(n, channel_count)):
        timestamp = float(i)
        b.append(Record(d, timestamp))

    rows = b.query(start=b.start_time)
    assert len(rows) == n
    assert len(b.all()) == n


def test_concurrent_access():
    """Test that queries to buffer still allow efficient writes."""

    channel_count = 25
    channels = ["ch" + str(c) for c in range(channel_count)]

    insert_timer = _Timer()

    buf = Buffer(channels=channels, chunksize=1000)
    channel_count = 25
    channels = ["ch" + str(c) for c in range(channel_count)]

    class Writer(_StoppableThread):
        def __init__(self, buf):
            super(Writer, self).__init__()
            self.buffer = buf

        def run(self):
            i = 0
            while self.running():
                data = [np.random.uniform(-1000, 1000)
                        for cc in range(channel_count)]
                with insert_timer:
                    self.buffer.append(Record(data, i))
                time.sleep(0.002)
                i += 1

    class Reader(_StoppableThread):
        def __init__(self, buf):
            super(Reader, self).__init__()
            self.buffer = buf
            self.records = []

        def run(self):
            while self.running():
                self.records = self.buffer.latest()
                time.sleep(0.1)

    write_thread = Writer(buf)
    read_thread = Reader(buf)

    write_thread.start()
    read_thread.start()

    time.sleep(1)

    write_thread.stop()
    read_thread.stop()

    assert len(insert_timer.times) > 100
    assert len(read_thread.records) > 0 and len(read_thread.records) <= 1000
    assert sum(insert_timer.times) / len(insert_timer.times) < 1 / 600
