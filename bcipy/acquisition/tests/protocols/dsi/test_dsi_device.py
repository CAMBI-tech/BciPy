"""Tests for DSI headset device driver."""
import unittest
import pytest
from bcipy.acquisition.datastream import tcp_server
from bcipy.acquisition.protocols.dsi.dsi_connector import DsiConnector
from bcipy.acquisition.protocols.dsi.dsi_protocol import DsiProtocol
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.connection_method import ConnectionMethod

HOST = '127.0.0.1'
DEFAULT_PORT = 9000

DEVICE_NAME = 'DSI'
DEVICE_CHANNELS = [
    "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "A1", "Fp1", "Fp2", "T3",
    "T5", "O1", "O2", "F7", "F8", "A2", "T6", "T4", "TRG"
]
DEVICE_SAMPLE_RATE = 300.0


def spec(name=DEVICE_NAME,
         channels=DEVICE_CHANNELS,
         sample_rate=DEVICE_SAMPLE_RATE):
    """Creates a DeviceSpec for testing purposes"""
    return DeviceSpec(name=name,
                      channels=channels,
                      sample_rate=sample_rate,
                      connection_methods=[ConnectionMethod.TCP])


class TestDsiDevice(unittest.TestCase):
    """Tests for DsiDevice"""

    def __init__(self, *args, **kwargs):
        super(TestDsiDevice, self).__init__(*args, **kwargs)
        self.host = HOST

    def connection_params(self):
        """Return connection params dict"""
        return {'host': self.host, 'port': self.port}

    @classmethod
    def setUpClass(cls):
        cls.server, cls.port = tcp_server.start_socket_server(
            DsiProtocol(device_spec=spec()), HOST, DEFAULT_PORT)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        """Run before each test."""

        self.port = type(self).port

    def test_mismatched_channels(self):
        """An exception should be thrown if parameters do not match data read
        from the device."""
        print("Running test_mismatched_channels")
        device = DsiConnector(connection_params=self.connection_params(),
                              device_spec=spec(channels=['ch1', 'ch2']))
        self.assertEqual(len(device.channels), 2)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_mismatched_frequency(self):
        """An exception should be thrown if parameters do not match data read
        from the device."""
        print("Running test_mismatched_frequency")
        device = DsiConnector(connection_params=self.connection_params(),
                              device_spec=spec(sample_rate=100.0))
        self.assertEqual(device.fs, 100)
        device.connect()

        with pytest.raises(Exception):
            device.acquisition_init()

    def test_read_data(self):
        """Should produce a valid sensor_data record."""

        print("Running test_read_data")
        device = DsiConnector(connection_params=self.connection_params(),
                              device_spec=spec())

        device.connect()
        device.acquisition_init()
        data = device.read_data()

        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), len(device.channels))
        for channel in data:
            self.assertTrue(isinstance(channel, float))


if __name__ == '__main__':
    unittest.main()
