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
import unittest


class _MockDevice(Device):
    """Device that mocks reading data. Does not need a server. Continues as
    long as there is mock data."""

    def __init__(self, data=[], channels=[], fs=500):
        super(_MockDevice, self).__init__(
            connection_params={}, channels=channels, fs=fs)
        self.data = data
        self.i = 0

    def name(self):
        return 'MockDevice'

    def read_data(self):
        if (self.i < len(self.data)):
            row = self.data[self.i]
            self.i += 1
            return row
        return None


class _MockProcessor(Processor):
    """Processor that doesn't do anything."""

    def process(self, record, timestamp=None):
        pass


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


class _CountingProcessor(Processor):
    """Processor that records all data passed to the process method."""

    def __init__(self):
        super(_CountingProcessor, self).__init__()
        self.data = []

    def process(self, record, timestamp=None):
        self.data.append(record)


class TestClient(unittest.TestCase):
    """Main Test class for client code."""

    def __init__(self, *args, **kwargs):
        super(TestClient, self).__init__(*args, **kwargs)
        num_channels = 25
        num_records = 500
        self.mock_channels = ['ch' + str(i) for i in range(num_channels)]
        self.mock_data = [[np.random.uniform(-1000, 1000)
                           for i in range(num_channels)]
                          for j in range(num_records)]

    def test_filewriter(self):
        """Test filewriter."""

        mwrite = mock_open()
        with patch('processor.open', mwrite):

            # Instantiate and start collecting data

            device = _MockDevice(data=self.mock_data,
                                 channels=self.mock_channels)
            daq = Client(device=device)
            with daq:
                time.sleep(0.1)

            mwrite.assert_called_once_with('rawdata.csv', 'wb')

            writeargs = [args[0]
                         for name, args, kwargs in mwrite().write.mock_calls]

            # First write was daq_type
            self.assertTrue('daq_type' in writeargs[0])
            self.assertTrue(device.name() in writeargs[0])

            # Second write was sample_rate
            self.assertTrue(writeargs[1].startswith('sample_rate,' +
                                                    str(device.fs)))

            # Third write was column header
            self.assertTrue(writeargs[2].startswith(
                ','.join(['timestamp'] + self.mock_channels)))

            self.assertTrue(daq.get_data_len() > 0,
                            "daq should have acquired data")

            # All subsequent data writes should look like data.
            for i, r in enumerate(writeargs[3:]):
                self.assertEqual(self.mock_data[i],
                                 [float(n) for n in r.split(",")[1:]])

            # Length of data writes should match the buffer size.
            self.assertEqual(daq.get_data_len(), len(writeargs[3:]))

            daq.cleanup()

    def test_processor(self):
        """Test processor calls."""

        device = _MockDevice(data=self.mock_data, channels=self.mock_channels)
        processor = _CountingProcessor()
        processor.set_device_info("MockDevice", 500, self.mock_channels)

        daq = Client(device=device, processor=processor)
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        self.assertTrue(len(processor.data) > 0)

        for i, record in enumerate(processor.data):
            self.assertEqual(record, self.mock_data[i])

        daq.cleanup()

    def test_buffer(self):
        """Buffer should capture values read from the device."""

        device = _MockDevice(data=self.mock_data, channels=self.mock_channels)
        daq = Client(device=device,
                     processor=_MockProcessor())
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        # Get all records from buffer
        data = daq.get_data()
        self.assertTrue(len(data) > 0, "Buffer should have data.")
        for i, record in enumerate(data):
            self.assertEqual(record.data, self.mock_data[i])

        daq.cleanup()

    def test_clock(self):
        """Test clock integration."""

        clock = _MockClock()
        daq = Client(device=_MockDevice(data=self.mock_data,
                                        channels=self.mock_channels),
                     processor=_MockProcessor(),
                     clock=clock)
        with daq:
            time.sleep(0.1)

        # Get all records from buffer
        data = daq.get_data()

        # NOTE: we can't make any assertions about the Clock, since it is
        # copied when it's passed to the acquisition thread.
        self.assertTrue(len(data) > 0)
        for i, record in enumerate(data):
            self.assertEqual(record.timestamp, float(i + 1))

        daq.cleanup()


if __name__ == '__main__':
    unittest.main()
