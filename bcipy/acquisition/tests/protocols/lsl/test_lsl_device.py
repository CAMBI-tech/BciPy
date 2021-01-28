# pylint: disable=no-self-use
"""Tests for the LabStreamingLayer driver."""

import unittest
import pytest
from bcipy.acquisition.datastream.lsl_server import LslDataServer, MARKER_STREAM_NAME
from bcipy.acquisition.datastream.tcp_server import await_start
from bcipy.acquisition.protocols.lsl.lsl_connector import LslConnector, LSL_TIMESTAMP
from bcipy.acquisition.devices import DeviceSpec


class TestLslDevice(unittest.TestCase):
    """Test the LslDevice"""

    def __init__(self, *args, **kwargs):
        super(TestLslDevice, self).__init__(*args, **kwargs)
        self.server = None
        self.device_spec = DeviceSpec(
            name='LSL',
            channels=['C3', 'C4', 'Cz', 'FPz', 'POz', 'CPz', 'O1', 'O2'],
            sample_rate=100)

    @property
    def channel_count(self):
        return self.device_spec.channel_count

    @property
    def channels(self):
        return self.device_spec.channels.copy()

    @property
    def sample_rate(self):
        return self.device_spec.sample_rate

    def default_data_server(self):
        return LslDataServer(device_spec=self.device_spec)

    def start_server(self, data_server: LslDataServer):
        self.server = data_server
        if self.server:
            await_start(self.server)

    def stop_server(self):
        if self.server:
            self.server.stop()
            self.server = None


class TestLslDeviceSpec(TestLslDevice):
    """LSL Device tests."""

    def setUp(self):
        """Run before each test."""
        self.start_server(self.default_data_server())

    def tearDown(self):
        """Run after each test."""
        self.stop_server()

    def test_incorrect_number_of_channels(self):
        """A list of channels with len that does not match channel_count should
        raise an exception."""

        spec = DeviceSpec(name='LSL',
                          channels=['ch1', 'ch2'],
                          sample_rate=self.sample_rate)
        device = LslConnector(connection_params={}, device_spec=spec)
        self.assertEqual(spec.channel_count, 2)
        self.assertNotEqual(len(device.channels), self.channel_count)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_device_info_channels(self):
        """Device should list the correct channels"""
        spec = DeviceSpec(name='LSL',
                          channels=self.channels,
                          sample_rate=self.sample_rate)
        device = LslConnector(connection_params={}, device_spec=spec)
        device.connect()
        device.acquisition_init()

        self.assertEqual(self.channels, device.channels)

    def test_incorrect_frequency(self):
        """Provided sample_rate should match sample rate read from device"""
        device = LslConnector(connection_params={},
                              device_spec=DeviceSpec(name='LSL',
                                                     channels=self.channels,
                                                     sample_rate=300))

        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_connect(self):
        """Should require a connect call before initialization."""
        device = LslConnector(connection_params={},
                              device_spec=DeviceSpec(
            name='LSL',
            channels=self.channels,
            sample_rate=self.sample_rate))

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_read_data(self):
        """Should produce a valid data record."""
        print(self.device_spec.channels)
        device = LslConnector(connection_params={},
                              device_spec=DeviceSpec(
            name='LSL',
            channels=self.channels,
            sample_rate=self.sample_rate))
        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))

        for channel in data[0:-1]:
            self.assertTrue(isinstance(channel, float))


class TestLslChannelConfig(TestLslDevice):
    """Test configuration options for LslDevice channels."""

    def __init__(self, *args, **kwargs):
        super(TestLslChannelConfig, self).__init__(*args, **kwargs)
        self.device_spec = DeviceSpec(name='DSI',
                                      channels=[
                                          'C3', 'C4', 'Cz', 'FPz', 'POz',
                                          'CPz', 'O1', 'O2', 'TRG'
                                      ],
                                      sample_rate=300)

    def tearDown(self):
        """Run after each test."""
        self.stop_server()

    def test_with_trigger_channel(self):
        """A device with a TRG channel should work as expected."""
        server = LslDataServer(device_spec=self.device_spec)
        self.start_server(server)
        device = LslConnector(connection_params={}, device_spec=self.device_spec)

        device.connect()
        device.acquisition_init()
        self.assertEquals(self.channels, device.channels)

    def test_with_marker_stream(self):
        server = LslDataServer(device_spec=self.device_spec, add_markers=True)
        self.start_server(server)
        device = LslConnector(connection_params={}, device_spec=self.device_spec)

        device.connect()
        device.acquisition_init()
        self.assertEquals(self.channels, device.channels)
        self.assertEquals(len(self.channels), len(device.read_data()))

    def test_with_marker_stream_included(self):
        server = LslDataServer(device_spec=self.device_spec, add_markers=True)
        self.start_server(server)
        device = LslConnector(connection_params={},
                              device_spec=self.device_spec,
                              include_marker_streams=True)

        device.connect()
        device.acquisition_init()
        self.assertEquals(self.channels + [MARKER_STREAM_NAME],
                          device.channels)
        self.assertEquals(len(self.channels) + 1, len(device.read_data()))

    def test_with_marker_stream_and_timestamp(self):
        server = LslDataServer(device_spec=self.device_spec, add_markers=True)
        self.start_server(server)
        device = LslConnector(connection_params={},
                              device_spec=self.device_spec,
                              include_lsl_timestamp=True,
                              include_marker_streams=True)

        device.connect()
        device.acquisition_init()
        self.assertEquals(self.channels + [LSL_TIMESTAMP, MARKER_STREAM_NAME],
                          device.channels)
        self.assertEquals(len(self.channels) + 2, len(device.read_data()))

    def test_renaming_columns(self):
        server = LslDataServer(device_spec=self.device_spec,
                               add_markers=True,
                               marker_stream_name='TRG')
        self.start_server(server)
        device = LslConnector(connection_params={},
                              device_spec=self.device_spec,
                              include_marker_streams=True,
                              rename_rules={'TRG': 'TRG_device_stream'})

        device.connect()
        device.acquisition_init()

        expected = [
            'C3', 'C4', 'Cz', 'FPz', 'POz', 'CPz', 'O1', 'O2',
            'TRG_device_stream', 'TRG'
        ]
        self.assertEquals(expected, device.channels)
        self.assertEquals(expected, device.device_info.channels)


if __name__ == '__main__':
    unittest.main()
