"""Unit tests for LslAcquisitionClient"""
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.devices import (IRREGULAR_RATE, DeviceSpec,
                                       preconfigured_device)
from bcipy.acquisition.protocols.lsl.lsl_client import (LslAcquisitionClient,
                                                        discover_device_spec)
from bcipy.helpers.clock import Clock

DEVICE_NAME = 'DSI-24'
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

    def test_discover_device_spec(self):
        """Test utility function for creating a DeviceSpec based on the
        content_type."""
        spec = discover_device_spec(content_type='EEG')
        self.assertEqual(spec.name, DEVICE.name)
        self.assertEqual(spec.sample_rate, DEVICE.sample_rate)
        self.assertListEqual(spec.channels, DEVICE.channels)

    def test_with_specified_device(self):
        """Test functionality with a provided device_spec"""
        client = LslAcquisitionClient(max_buffer_len=1, device_spec=DEVICE)
        self.assertEqual(client.max_samples, DEVICE.sample_rate)
        client.start_acquisition()
        time.sleep(1)
        samples = client.get_latest_data()
        client.stop_acquisition()
        # Delta values are usually much closer locally (1.0) but can vary
        # widely in continuous integration.
        self.assertAlmostEqual(int(DEVICE.sample_rate),
                               len(samples),
                               delta=5.0)

    def test_specified_device_wrong_channels(self):
        """Should throw an exception if channels don't match metadata."""
        client = LslAcquisitionClient(
            max_buffer_len=1, device_spec=preconfigured_device('DSI-VR300'))

        with self.assertRaises(Exception):
            client.start_acquisition()

    def test_specified_device_wrong_sample_rate(self):
        """Should throw an exception if sample_rate doesn't match metadata."""
        device = DeviceSpec(name=DEVICE.name,
                            channels=DEVICE.channels,
                            sample_rate=DEVICE.sample_rate + 100,
                            content_type=DEVICE.content_type,
                            data_type=DEVICE.data_type)
        client = LslAcquisitionClient(max_buffer_len=1, device_spec=device)
        with self.assertRaises(Exception):
            client.start_acquisition()

    def test_specified_device_for_markers(self):
        """Should allow a device_spec for a Marker stream."""
        device = DeviceSpec(name='Mouse',
                            channels=['Marker'],
                            sample_rate=IRREGULAR_RATE,
                            content_type='Markers',
                            data_type='string')

        client = LslAcquisitionClient(max_buffer_len=1024, device_spec=device)
        self.assertEqual(1024, client.max_samples)

    def test_with_unspecified_device(self):
        """Test with unspecified device."""
        client = LslAcquisitionClient(max_buffer_len=1)
        client.start_acquisition()

        self.assertEqual(DEVICE.name, client.device_spec.name)
        self.assertEqual(DEVICE.channels, client.device_spec.channels)
        self.assertEqual(DEVICE.sample_rate, client.device_spec.sample_rate)
        time.sleep(1)
        samples = client.get_latest_data()
        client.stop_acquisition()
        self.assertAlmostEqual(DEVICE.sample_rate, len(samples), delta=5.0)

    @pytest.mark.slow
    def test_get_data(self):
        """Test functionality with a provided device_spec"""
        client = LslAcquisitionClient(max_buffer_len=1, device_spec=DEVICE)
        client.start_acquisition()

        experiment_clock = Clock(start_at_zero=True)
        # Ensure we are at 0
        experiment_clock.reset()
        time.sleep(1)

        # Get a half second of data
        offset = client.clock_offset(experiment_clock)
        start = 0.5 + offset
        samples = client.get_data(start, limit=100)
        samples2 = client.get_data(start, limit=100)

        client.stop_acquisition()
        self.assertEqual(100, len(samples))
        self.assertEqual(samples, samples2,
                         "Consecutive queries should yield the same answers")
        self.assertAlmostEqual(client.convert_time(experiment_clock, 0.5),
                               start,
                               delta=0.002)

    @pytest.mark.slow
    def test_event_offset(self):
        """Test the offset in seconds of a given event relative to the first
        sample time."""
        client = LslAcquisitionClient(max_buffer_len=1, device_spec=DEVICE)
        experiment_clock = Clock(start_at_zero=True)

        client.start_acquisition()

        # Ensure experiment clock is at 0
        experiment_clock.reset()

        # We don't need to wait for any data. Starting acquisition pulls a sample.
        client.stop_acquisition()

        event_time = 0.5
        self.assertAlmostEqual(client.event_offset(experiment_clock,
                                                   event_time),
                               0.5,
                               delta=0.02)

    def test_with_recording(self):
        """Test that recording works."""

        temp_dir = tempfile.mkdtemp()
        filename = f'eeg_data_{DEVICE_NAME.lower()}.csv'
        path = Path(temp_dir, filename)

        self.assertFalse(path.exists())

        client = LslAcquisitionClient(max_buffer_len=1,
                                      save_directory=temp_dir,
                                      raw_data_file_name=filename)
        client.start_acquisition()
        time.sleep(0.1)
        client.stop_acquisition()
        self.assertTrue(path.exists())

        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
