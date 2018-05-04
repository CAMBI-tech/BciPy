
import time
from random import random as rand

import pytest
from acquisition.util import StoppableThread
from protocols.lsl.lsl_device import LslDevice
from pylsl import StreamInfo, StreamOutlet
import unittest

class Server(StoppableThread):
    """LSL Server"""

    def __init__(self, params={}, include_meta=False):
        super(Server, self).__init__()

        self.channel_count = params['channel_count']
        info = StreamInfo('TestStream', 'EEG', params['channel_count'],
                          params['hz'], 'float32', 'uid12345')

        if include_meta:
            meta_channels = info.desc().append_child('channels')
            for c in params['channels']:
                meta_channels.append_child('channel') \
                    .append_child_value('label', c) \
                    .append_child_value('unit', 'microvolts') \
                    .append_child_value('type', 'EEG')

        self.outlet = StreamOutlet(info)

    def stop(self):
        super(Server, self).stop()
        # del self.outlet
        # Allows pylsl to cleanup; The outlet will no longer be discoverable
        # after destruction and all connected inlets will stop delivering data.
        self.outlet = None

    def random_sample(self):
        return [rand() for k in range(self.channel_count)]

    def run(self):

        while self.running():
            self.outlet.push_sample(self.random_sample())
            time.sleep(0.01)


class TestLslDevice(unittest.TestCase):
    """Test the LslDevice"""

    def __init__(self, *args, **kwargs):
        super(TestLslDevice, self).__init__(*args, **kwargs)
        self.channels = ['C3', 'C4', 'Cz', 'FPz', 'POz', 'CPz', 'O1', 'O2']
        self.channel_count = len(self.channels)
        self.hz = 100
        self.server_params = {'channel_count': self.channel_count,
                              'channels': self.channels,
                              'hz': self.hz}

    @property
    def include_meta(self):
        raise Exception("Must be implemented in subclass")

    def make_server(self):
        # TODO: subclass for different server params
        return Server(include_meta=self.include_meta,
                      params=self.server_params)

    def setUp(self):
        """Run before each test."""
        self.server = self.make_server()
        self.server.start()

    def tearDown(self):
        """Run after each test."""
        self.server.stop()
        self.server = None


class TestLslWithoutMetadata(TestLslDevice):
    """LSL Device tests in which the server does not provide metadata."""

    @property
    def include_meta(self):
        return False

    def test_incorrect_number_of_channels(self):
        """A list of channels with len that does not match channel_count should
        raise an exception."""

        device = LslDevice(connection_params={}, fs=self.hz,
                           channels=['ch1', 'ch2'])
        self.assertEqual(len(device.channels), 2)
        self.assertNotEqual(len(device.channels), self.channel_count)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_incorrect_frequency(self):
        """Provided fs should match sample rate read from device"""
        device = LslDevice(connection_params={},
                           channels=self.channels, fs=300)
        self.assertEqual(device.fs, 300)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_frequency_init(self):
        """fs should be initialized from device metadata if not provided"""

        device = LslDevice(connection_params={},
                           channels=self.channels, fs=None)
        device.connect()
        device.acquisition_init()

        self.assertEqual(device.fs, self.hz)

    def test_connect(self):
        """Should require a connect call before initialization."""
        device = LslDevice(connection_params={},
                           channels=[], fs=None)

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_read_data(self):
        """Should produce a valid data record."""

        device = LslDevice(connection_params={},
                           channels=self.channels, fs=self.hz)

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for f in data:
            self.assertTrue(isinstance(f, float))


class TestLslWithMetadata(TestLslDevice):
    """LSL Device tests in which the server provides metadata."""

    @property
    def include_meta(self):
        return True

    def test_mismatched_channel_names(self):
        """Provided channel names should match device information."""
        channels = ['ch' + str(i) for i in range(self.channel_count)]
        device = LslDevice(connection_params={}, fs=self.hz, channels=channels)

        self.assertEqual(len(device.channels), self.channel_count)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def _test_channel_init(self):
        """Channels should be initialized from device metadata if not
         provided."""
        device = LslDevice(connection_params={}, fs=self.hz,
                           channels=[])
        self.assertEqual(len(device.channels), 0)
        device.connect()
        device.acquisition_init()
        self.assertEqual(len(device.channels), self.channel_count)
        self.assertEqual(device.channels, self.channels)

    def test_frequency_init(self):
        """fs should be initialized from device metadata if not provided"""

        device = LslDevice(connection_params={},
                           channels=self.channels, fs=None)
        device.connect()
        device.acquisition_init()

        self.assertEqual(device.fs, self.hz)

    def test_connect(self):
        """Should require a connect call before initialization."""
        device = LslDevice(connection_params={},
                           channels=[], fs=None)

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_read_data(self):
        """Should produce a valid data record."""

        device = LslDevice(connection_params={},
                           channels=self.channels, fs=self.hz)

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for f in data:
            self.assertTrue(isinstance(f, float))


if __name__ == '__main__':
    unittest.main()
