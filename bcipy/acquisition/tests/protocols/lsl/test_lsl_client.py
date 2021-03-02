"""Unit tests for LslAcquisitionClient"""
import time
import unittest
from typing import List
from bcipy.acquisition.connection_method import ConnectionMethod
from bcipy.acquisition.devices import preconfigured_device, DeviceSpec, IRREGULAR_RATE
from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.protocols.lsl.lsl_client import LslAcquisitionClient
from mockito import mock

DEVICE_NAME = 'DSI'
DEVICE = preconfigured_device(DEVICE_NAME)


class TestDataAcquisitionClient(unittest.TestCase):
    """Main Test class for DataAcquisitionClient code."""

    def __init__(self, *args, **kwargs):
        super(TestDataAcquisitionClient, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        """Set up a mock LSL data server that is streaming for all tests."""
        cls.eeg_server = LslDataServer(device_spec=DEVICE)
        cls.eeg_server.start()

    @classmethod
    def tearDownClass(cls):
        """"Stop the data server."""
        cls.eeg_server.stop()

    def test_with_specified_device(self):
        """Test functionality with a provided device_spec"""
        client = LslAcquisitionClient(max_buflen=1, device_spec=DEVICE)
        client.start_acquisition()
        time.sleep(1)
        samples = client.get_latest_data()
        client.stop_acquisition()
        self.assertEqual(DEVICE.sample_rate, len(samples))

    def test_specified_device_wrong_channels(self):
        """Should throw an exception if channels don't match metadata."""
        client = LslAcquisitionClient(max_buflen=1,
                                      device_spec=preconfigured_device('LSL'))

        with self.assertRaises(Exception):
            client.start_acquisition()

    def test_specified_device_wrong_sample_rate(self):
        """Should throw an exception if sample_rate doesn't match metadata."""
        device = DeviceSpec(name=DEVICE.name,
                            channels=DEVICE.channels,
                            sample_rate=DEVICE.sample_rate + 100,
                            content_type=DEVICE.content_type,
                            data_type=DEVICE.data_type,
                            connection_methods=DEVICE.connection_methods)
        client = LslAcquisitionClient(max_buflen=1, device_spec=device)
        with self.assertRaises(Exception):
            client.client.start_acquisition()

    def test_specified_device_wrong_connection_type(self):
        """Should only allow devices that can connect to LSL"""
        device = DeviceSpec(name=DEVICE.name,
                            channels=DEVICE.channels,
                            sample_rate=DEVICE.sample_rate,
                            content_type=DEVICE.content_type,
                            data_type=DEVICE.data_type,
                            connection_methods=[ConnectionMethod.TCP])
        with self.assertRaises(Exception):
            client = LslAcquisitionClient(max_buflen=1, device_spec=device)
    
    def test_specified_device_for_markers(self):
        """Should not allow a device_spec for a Marker stream."""
        device = DeviceSpec(name='Mouse',
                            channels=['Btn1', 'Btn2'],
                            sample_rate=IRREGULAR_RATE,
                            content_type='Markers',
                            data_type='string')
        with self.assertRaises(Exception):
            client = LslAcquisitionClient(max_buflen=1, device_spec=device)

    def test_with_unspecified_device(self):
        """Test with unspecified device."""
        client = LslAcquisitionClient(max_buflen=1)
        client.start_acquisition()

        self.assertEquals(DEVICE.name, client.device_spec.name)
        self.assertEquals(DEVICE.channels, client.device_spec.channels)
        self.assertEquals(DEVICE.sample_rate, client.device_spec.sample_rate)
        time.sleep(1)
        samples = client.get_latest_data()
        client.stop_acquisition()
        self.assertEqual(DEVICE.sample_rate, len(samples))

    def test_with_recording(self):
        """Test that recording works."""
        # TODO:
        pass


if __name__ == '__main__':
    unittest.main()