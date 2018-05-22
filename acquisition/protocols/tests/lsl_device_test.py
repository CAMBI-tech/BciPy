

import pytest
from acquisition.datastream.lsl_server import LslDataServer
from acquisition.datastream.server import await_start
from acquisition.datastream import generator
from protocols.lsl.lsl_device import LslDevice
import unittest


class TestLslDevice(unittest.TestCase):
    """Test the LslDevice"""

    def __init__(self, *args, **kwargs):
        super(TestLslDevice, self).__init__(*args, **kwargs)
        self.channels = ['C3', 'C4', 'Cz', 'FPz', 'POz', 'CPz', 'O1', 'O2']
        self.channel_count = len(self.channels)
        self.hz = 100

    @property
    def include_meta(self):
        raise Exception("Must be implemented in subclass")

    def setUp(self):
        """Run before each test."""
        self.server = LslDataServer(params={'name': 'LSL',
                                            'channels': self.channels,
                                            'hz': self.hz},
                                    generator=generator.random_data(
                                        channel_count=self.channel_count),
                                    include_meta=self.include_meta,
                                    add_markers=True)
        await_start(self.server)

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
                           channels=self.channels + ["TRG"], fs=None)
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
                           channels=self.channels + ["TRG"], fs=self.hz)

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))

        for f in data[0:-1]:
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
        self.assertEqual(len(device.channels), self.channel_count + 1,
                         "Should have added a TRG channel")
        self.assertEqual(device.channels[0:-1], self.channels)

    def test_frequency_init(self):
        """fs should be initialized from device metadata if not provided"""

        device = LslDevice(connection_params={}, fs=None)
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

        device = LslDevice(connection_params={}, fs=self.hz)

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for f in data[0:-1]:
            self.assertTrue(isinstance(f, float))


if __name__ == '__main__':
    unittest.main()
