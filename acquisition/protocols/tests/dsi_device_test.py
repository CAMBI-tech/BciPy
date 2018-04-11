from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
from datastream import generator, server
from protocols.dsi.dsi_device import dsi
from protocols.dsi.dsi_device import DsiDevice
from protocols.dsi.dsi_protocol import DsiProtocol

import unittest

HOST = '127.0.0.1'
PORT = 9999
connection_params = {'host': HOST, 'port': PORT}


def make_server():
    protocol = DsiProtocol()
    channel_count = len(protocol.channels)
    return server.DataServer(protocol=protocol,
                             generator=generator.random_data,
                             gen_params={'channel_count': channel_count},
                             host=HOST, port=PORT)


class TestDsiDevice(unittest.TestCase):
    """Tests for DsiDevice"""

    def test_device(self):
        s = make_server()
        s.start()

        self._acquisition_init()
        self._connect()
        self._read_data()

        s.stop()

    def test_channels(self):
        """An exception should be thrown if parameters do not match data read
        from the device."""

        s = make_server()
        s.start()

        device = DsiDevice(connection_params=connection_params,
                           channels=['ch1', 'ch2'])
        self.assertEqual(len(device.channels), 2)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

        s.stop()

    def test_frequency(self):
        """An exception should be thrown if parameters do not match data read
        from the device."""
        s = make_server()
        s.start()

        device = DsiDevice(connection_params=connection_params, fs=100)
        self.assertEqual(device.fs, 100)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

        s.stop()

    def _acquisition_init(self):
        """Channel and sample rate properties should be updated by reading
        initialization data from the server."""

        device = DsiDevice(connection_params=connection_params, channels=[])
        self.assertEqual(device.fs, dsi.DEFAULT_FS)
        self.assertEqual(len(device.channels), 0)

        device.connect()
        device.acquisition_init()

        self.assertEqual(device.fs, dsi.DEFAULT_FS)
        self.assertEqual(len(device.channels), len(dsi.DEFAULT_CHANNELS))

    def _connect(self):
        """Should require a connect call before initialization."""

        device = DsiDevice(connection_params=connection_params)

        # Payload size that exceeds sensor_data points should throw an error
        with pytest.raises(AssertionError):
            device.acquisition_init()

    def _read_data(self):
        """Should produce a valid sensor_data record."""

        device = DsiDevice(connection_params=connection_params)

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for f in data:
            self.assertTrue(isinstance(f, float))
