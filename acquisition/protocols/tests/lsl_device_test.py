from __future__ import absolute_import, division, print_function

import time
from random import random as rand

import pytest
from acquisition.util import StoppableThread
from protocols.lsl.lsl_device import LslDevice
from pylsl import StreamInfo, StreamOutlet
import unittest

CHANNEL_COUNT = 8
CHANNELS = ['C3', 'C4', 'Cz', 'FPz', 'POz', 'CPz', 'O1', 'O2']
HZ = 100


class Server(StoppableThread):
    '''docstring for Server'''

    def __init__(self, include_meta=False):
        super(Server, self).__init__()

        info = StreamInfo('TestStream', 'EEG', CHANNEL_COUNT,
                          HZ, 'float32', 'uid12345')

        if include_meta:
            channels = info.desc().append_child('channels')
            for c in CHANNELS:
                channels.append_child('channel') \
                    .append_child_value('label', c) \
                    .append_child_value('unit', 'microvolts') \
                    .append_child_value('type', 'EEG')
        self.outlet = StreamOutlet(info)
    # @override ; context manager

    def __enter__(self):
        self.start()
        return self

    # @override ; context manager
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def run(self):
        while self.running():
            sample = [rand() for k in range(CHANNEL_COUNT)]
            self.outlet.push_sample(sample)
            time.sleep(0.1)


class TestLslDevice(unittest.TestCase):
    """Test the LslDevice"""

    def test_minimal_metadata_invariants(self):
        """Run tests with a server with minimal metadata, missing channel
         names."""
        with Server(include_meta=False):
            self._test_incorrect_number_of_channels()
            self._test_incorrect_frequency()
            self._test_frequency_init()
            self._test_connect()
            self._test_read_data()

    def test_full_metadata_invariants(self):
        """Run tests with a server with full metadata including channel
         names."""
        with Server(include_meta=True):
            self._test_mismatched_channel_names()
            self._test_channel_init()
            self._test_frequency_init()
            self._test_connect()
            self._test_read_data()

    def _test_incorrect_number_of_channels(self):
        """A list of channels with len that does not match channel_count should
        raise an exception."""

        device = LslDevice(connection_params={}, fs=HZ,
                           channels=['ch1', 'ch2'])
        self.assertEqual(len(device.channels), 2)
        self.assertNotEqual(len(device.channels), CHANNEL_COUNT)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def _test_incorrect_frequency(self):
        """Provided fs should match sample rate read from device"""
        device = LslDevice(connection_params={},
                           channels=CHANNELS, fs=300)
        self.assertEqual(device.fs, 300)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def _test_frequency_init(self):
        """fs should be initialized from device metadata if not provided"""

        device = LslDevice(connection_params={},
                           channels=CHANNELS, fs=None)
        device.connect()
        device.acquisition_init()

        self.assertEqual(device.fs, HZ)

    def _test_connect(self):
        """Should require a connect call before initialization."""
        device = LslDevice(connection_params={},
                           channels=[], fs=None)

        with pytest.raises(Exception):
            device.acquisition_init()

    def _test_read_data(self):
        """Should produce a valid data record."""

        device = LslDevice(connection_params={},
                           channels=CHANNELS, fs=HZ)

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for f in data:
            self.assertTrue(isinstance(f, float))

    def _test_mismatched_channel_names(self):
        """Provided channel names should match device information."""
        channels = ['ch' + str(i) for i in range(CHANNEL_COUNT)]
        device = LslDevice(connection_params={}, fs=HZ, channels=channels)

        self.assertEqual(len(device.channels), CHANNEL_COUNT)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def _test_channel_init(self):
        """Channels should be initialized from device metadata if not
         provided."""
        device = LslDevice(connection_params={}, fs=HZ,
                           channels=[])
        self.assertEqual(len(device.channels), 0)
        device.connect()
        device.acquisition_init()
        self.assertEqual(len(device.channels), CHANNEL_COUNT)
        self.assertEqual(device.channels, CHANNELS)
