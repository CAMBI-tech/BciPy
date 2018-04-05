"""Functions to generate EEG(-like) data for testing and development.
Generators are used by a Producer to stream the data at a given frequency.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time

import numpy as np
from acquisition.client import Client
from acquisition.processor import Processor
from acquisition.protocols.device import Device
from mock import mock_open, patch


NUM_CHANNELS = 25
NUM_RECORDS = 500
MOCK_CHANNELS = ['ch' + str(i) for i in range(NUM_CHANNELS)]
MOCK_DATA = [[np.random.uniform(-1000, 1000) for i in range(NUM_CHANNELS)]
             for j in range(NUM_RECORDS)]


class _MockDevice(Device):
    """Device that mocks reading data. Does not need a server. Continues as
    long as there is mock data."""

    def __init__(self, fs=500):
        super(_MockDevice, self).__init__(
            connection_params={}, channels=MOCK_CHANNELS, fs=fs)
        self.i = 0

    def name(self):
        return 'MockDevice'

    def read_data(self):
        if (self.i < len(MOCK_DATA)):
            row = MOCK_DATA[self.i]
            self.i += 1
            return row
        return None


class _MockProcessor(Processor):
    """Processor that doesn't do anything."""

    def process(self, record, timestamp=None):
        pass


def test_filewriter():
    """Test filewriter."""

    mwrite = mock_open()
    with patch('processor.open', mwrite):

        # Instantiate and start collecting data

        device = _MockDevice()
        daq = Client(device=device)
        with daq:
            time.sleep(0.1)

        mwrite.assert_called_once_with('rawdata.csv', 'wb')

        writeargs = [args[0]
                     for name, args, kwargs in mwrite().write.mock_calls]

        # First write was daq_type
        assert 'daq_type' in writeargs[0]
        assert device.name() in writeargs[0]

        # Second write was sample_rate
        assert writeargs[1].startswith('sample_rate,' + str(device.fs))

        # Third write was column header
        assert writeargs[2].startswith(
            ','.join(['timestamp'] + MOCK_CHANNELS))

        assert daq.get_data_len() > 0, "daq should have acquired data"

        # All subsequent data writes should look like data.
        for i, r in enumerate(writeargs[3:]):
            assert MOCK_DATA[i] == [float(n) for n in r.split(",")[1:]]

        # Length of data writes should match the buffer size.
        assert daq.get_data_len() == len(writeargs[3:])

        daq.cleanup()


def test_processor():
    """Test processor calls."""

    class _CountingProcessor(Processor):
        """Processor that records all data passed to the process method."""

        def __init__(self):
            super(_CountingProcessor, self).__init__()
            self.data = []

        def process(self, record, timestamp=None):
            self.data.append(record)

    device = _MockDevice()
    processor = _CountingProcessor()
    processor.set_device_info("MockDevice", 500, MOCK_CHANNELS)

    daq = Client(device=device, processor=processor)
    daq.start_acquisition()
    time.sleep(0.1)
    daq.stop_acquisition()

    assert len(daq._processor.data) > 0

    for i, record in enumerate(daq._processor.data):
        assert record == MOCK_DATA[i]

    daq.cleanup()


def test_buffer():
    """Buffer should capture values read from the device."""

    device = _MockDevice()
    daq = Client(device=device,
                 processor=_MockProcessor())
    daq.start_acquisition()
    time.sleep(0.1)
    daq.stop_acquisition()

    # Get all records from buffer
    data = daq.get_data()
    assert len(data) > 0, "Data should have been put in the buffer."
    for i, record in enumerate(data):
        assert record.data == MOCK_DATA[i]

    daq.cleanup()


def test_clock():
    """Test clock integration."""

    class _MockClock(object):
        """Clock that provides timestamp values starting at 1.0; the next value
        is the increment of the previous."""

        def __init__(self):
            super(_MockClock, self).__init__()
            self.counter = 0

        def reset(self):
            self.counter = 0

        def getTime(self):
            self.counter += 1
            return float(self.counter)

    clock = _MockClock()
    daq = Client(device=_MockDevice(),
                 processor=_MockProcessor(),
                 clock=clock)
    with daq:
        time.sleep(0.1)

    # Get all records from buffer
    data = daq.get_data()

    # NOTE: we can't make any assertions about the Clock, since it is copied
    # when it's passed to the acquisition thread.
    assert len(data) > 0
    for i, record in enumerate(data):
        assert record.timestamp == float(i + 1)

    daq.cleanup()
