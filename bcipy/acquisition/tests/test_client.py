"""Functions to generate EEG(-like) data for testing and development.
Generators are used by a Producer to stream the data at a given frequency.
"""

import time

import numpy as np
from bcipy.acquisition.client import Client
from bcipy.acquisition.processor import Processor
from bcipy.acquisition.protocols.device import Device
from mock import mock_open, patch
import multiprocessing
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


class _AccumulatingProcessor(Processor):
    """Processor that records all data passed to the process method."""

    def __init__(self, q):
        super(_AccumulatingProcessor, self).__init__()
        self.data = q

    def process(self, record, timestamp=None):
        self.data.put(record)


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

    # def test_processor(self):
    #     """Test processor calls."""

    #     device = _MockDevice(data=self.mock_data, channels=self.mock_channels)
    #     q = multiprocessing.Queue()
    #     processor = _AccumulatingProcessor(q)

    #     daq = Client(device=device, processor=processor)
    #     daq.start_acquisition()
    #     time.sleep(0.1)
    #     daq.stop_acquisition()

    #     q.put('exit')
    #     data = []
    #     for d in iter(q.get, 'exit'):
    #         data.append(d)

    #     self.assertTrue(len(data) > 0)

    #     for i, record in enumerate(data):
    #         self.assertEqual(record, self.mock_data[i])

    #     daq.cleanup()

    def test_get_data(self):
        """Data should be queryable."""

        device = _MockDevice(data=self.mock_data, channels=self.mock_channels)
        daq = Client(device=device,
                     processor=_MockProcessor())
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        # Get all records
        data = daq.get_data()
        self.assertTrue(len(data) > 0, "Client should have data.")
        for i, record in enumerate(data):
            self.assertEqual(record.data, self.mock_data[i])

        daq.cleanup()

    def test_clock(self):
        """Test clock integration."""

        clock = _MockClock()
        clock.counter = 10  # ensures that clock gets reset.
        daq = Client(device=_MockDevice(data=self.mock_data,
                                        channels=self.mock_channels),
                     processor=_MockProcessor(),
                     buffer_name='buffer_client_test_clock.db',
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

    def test_offset(self):
        """Test offset calculation"""

        channels = ['ch1', 'ch2', 'TRG']
        fs = 100
        trigger_at = 10
        num_records = 500
        num_channels = len(channels)

        mock_data = []
        for i in range(num_records):
            d = [np.random.uniform(-100, 100) for _ in range(num_channels - 1)]
            trigger_channel = 0 if (i + 1) < trigger_at else 1
            d.append(trigger_channel)
            mock_data.append(d)

        device = _MockDevice(data=mock_data, channels=channels, fs=fs)
        daq = Client(device=device,
                     processor=_MockProcessor(),
                     buffer_name='buffer_client_test_offset.db',
                     clock=_MockClock())

        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        # The assertions should work before the stop_acquisition, but on some
        # Windows environments the tests were taking too long to setup and the
        # time would complete before any data had been processed.
        self.assertTrue(daq.is_calibrated)
        self.assertEqual(daq.offset, float(trigger_at) / fs)

        daq.cleanup()

    def test_missing_offset(self):

        channels = ['ch1', 'ch2', 'TRG']
        fs = 100
        trigger_at = 10
        num_records = 500
        num_channels = len(channels)

        mock_data = []
        for i in range(num_records):
            d = [np.random.uniform(-100, 100) for _ in range(num_channels - 1)]
            d.append(0)
            mock_data.append(d)

        device = _MockDevice(data=mock_data, channels=channels, fs=fs)
        daq = Client(device=device,
                     processor=_MockProcessor(),
                     clock=_MockClock(),
                     buffer_name='buffer_client_test_missing_offset.db')

        with daq:
            time.sleep(0.1)

        self.assertFalse(daq.is_calibrated)
        self.assertEqual(daq.offset, None)
        daq.cleanup()

    def test_zero_offset(self):
        """Test offset value override"""

        channels = ['ch1', 'ch2', 'TRG']
        fs = 100
        trigger_at = 10
        num_records = 500
        num_channels = len(channels)

        mock_data = []
        for i in range(num_records):
            d = [np.random.uniform(-100, 100) for _ in range(num_channels - 1)]
            trigger_channel = 0 if (i + 1) < trigger_at else 1
            d.append(trigger_channel)
            mock_data.append(d)

        device = _MockDevice(data=mock_data, channels=channels, fs=fs)
        daq = Client(device=device,
                     processor=_MockProcessor(),
                     buffer_name='buffer_client_test_offset.db',
                     clock=_MockClock())

        daq.is_calibrated = True  # force the override.
        daq.start_acquisition()
        time.sleep(0.1)
        daq.stop_acquisition()

        self.assertTrue(daq.is_calibrated)
        self.assertEqual(daq.offset, 0.0, "Setting the is_calibrated to True\
            should override the offset calcution.")

        daq.cleanup()


if __name__ == '__main__':
    unittest.main()
