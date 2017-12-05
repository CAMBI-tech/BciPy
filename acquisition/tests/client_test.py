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


class _MockProcessor(Processor):
    """Processor that doesn't do anything."""

    def __init__(self, device_name, fs, channels):
        super(_MockProcessor, self).__init__(device_name, fs, channels)

    def process(self, record, timestamp=None):
        pass


class _MockDevice(Device):
    """Device that mocks reading data every 1/500 seconds. Does not need a
    server. Accumulates read data in a list."""

    def __init__(self, channels, fs=500):
        super(_MockDevice, self).__init__(
            connection_params={}, channels=channels, fs=fs)
        self.data = []
        self._connected = True

    def name(self):
        return 'MockDevice'

    def read_data(self):
        time.sleep(1 / self.fs)
        row = [np.random.uniform(-1000, 1000)
               for i in range(len(self.channels))]
        if self._connected:
            self.data.append(row)
        return row

    def disconnect(self):
        self._connected = False


def test_filewriter():
    """Test filewriter."""

    mwrite = mock_open()
    with patch('processor.open', mwrite):

        # Instantiate and start collecting data

        channels = ['ch' + str(i) for i in range(25)]
        device = _MockDevice(channels)
        daq = Client(device=device)
        with daq:
            time.sleep(0.5)

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
            ','.join(['timestamp'] + device.channels))

        # All subsequent data writes should match the values in the data
        # rows.
        for i, r in enumerate(writeargs[3:]):
            assert repr(device.data[i][0]) in r

        # Length of data writes should match the buffer size.
        assert daq.get_data_len() == len(writeargs[3:])

        daq.cleanup()

def test_processor():
    """Test processor calls."""

    class _CountingProcessor(Processor):
        """Processor that records all data passed to the process method."""

        def __init__(self, device_name, fs, channels):
            super(_CountingProcessor, self).__init__(device_name, fs, channels)
            self.data = []

        def process(self, record, timestamp=None):
            self.data.append(record)

    device = _MockDevice(channels=['ch' + str(i) for i in range(10)])
    daq = Client(device=device, processor=_CountingProcessor)
    daq.start_acquisition()
    time.sleep(0.1)
    daq.stop_acquisition()

    assert len(daq._processor.data) > 0
    assert daq.get_data_len() == len(daq._processor.data), \
        'Processor should be called for every data read'

    for i, record in enumerate(daq._processor.data):
        assert record == device.data[i]

    daq.cleanup()

def test_buffer():
    """Buffer should capture values read from the device."""

    device = _MockDevice(channels=['ch' + str(i) for i in range(10)])
    daq = Client(device=device,
                 processor=_MockProcessor)
    daq.start_acquisition()
    time.sleep(0.1)
    daq.stop_acquisition()

    # Get all records from buffer
    data = daq.get_data()
    assert len(data) == len(device.data)
    for i, record in enumerate(data):
        assert record.data == device.data[i]

    daq.cleanup()


def test_clock():
    """Test clock integration."""

    class _MockClock(object):
        """Clock that provides timestamp values starting at 1.0; the next value
        is the increment of the previous."""

        def __init__(self):
            super(_MockClock, self).__init__()
            self.counter = 0

        def getTime(self):
            self.counter += 1
            return float(self.counter)

    clock = _MockClock()
    channels = ['ch' + str(i) for i in range(5)]
    daq = Client(device=_MockDevice(channels),
                 processor=_MockProcessor,
                 clock=clock)
    with daq:
        time.sleep(0.1)

    # Get all records from buffer
    data = daq.get_data()

    assert clock.counter > 0
    assert len(data) == clock.counter
    for i in range(clock.counter):
        assert data[i].timestamp == float(i + 1)

    daq.cleanup()