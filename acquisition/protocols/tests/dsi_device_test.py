from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
from datastream import generator, server
from protocols.dsi.dsi_device import dsi
from protocols.dsi.dsi_device import DsiDevice
from protocols.dsi.dsi_protocol import DsiProtocol
from util import StoppableThread

import unittest


class TestDsiDevice(unittest.TestCase):
    """Tests for DsiDevice"""

    def __init__(self, *args, **kwargs):
        super(TestDsiDevice, self).__init__(*args, **kwargs)
        self.host = '127.0.0.1'
        self.port = 9000

    def connection_params(self):
        return {'host': self.host, 'port': self.port}

    def setUp(self):
        """Run before each test."""

        protocol = DsiProtocol()
        channel_count = len(protocol.channels)
        self.server, self.port = server.start_socket_server(
            protocol, self.host, self.port)

    def tearDown(self):
        """Run after each test."""
        self.server.stop()
        self.server = None

    def test_mismatched_channels(self):
        """An exception should be thrown if parameters do not match data read
        from the device."""

        device = DsiDevice(connection_params=self.connection_params(),
                           channels=['ch1', 'ch2'])
        self.assertEqual(len(device.channels), 2)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_mismatched_frequency(self):
        """An exception should be thrown if parameters do not match data read
        from the device."""

        device = DsiDevice(connection_params=self.connection_params(), fs=100)
        self.assertEqual(device.fs, 100)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_update_params_on_init(self):
        """Channel and sample rate properties should be updated by reading
        initialization data from the server."""

        device = DsiDevice(
            connection_params=self.connection_params(), channels=[])
        self.assertEqual(device.fs, dsi.DEFAULT_FS)
        self.assertEqual(len(device.channels), 0)

        device.connect()
        device.acquisition_init()

        self.assertEqual(device.fs, dsi.DEFAULT_FS)
        self.assertEqual(len(device.channels), len(dsi.DEFAULT_CHANNELS))

    def test_read_data(self):
        """Should produce a valid sensor_data record."""

        device = DsiDevice(connection_params=self.connection_params())

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for f in data:
            self.assertTrue(isinstance(f, float))


if __name__ == '__main__':
    unittest.main()
