"""Tests for LslRecorder"""
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.datastream.mock.eye_tracker_server import eye_tracker_server
from bcipy.acquisition.devices import preconfigured_device
from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecorder
from bcipy.helpers.raw_data import TIMESTAMP_COLUMN, load

DEVICE_NAME = 'DSI-24'
DEVICE = preconfigured_device(DEVICE_NAME)


class TestLslRecorder(unittest.TestCase):
    """Main Test class for LslRecorder code."""

    def __init__(self, *args, **kwargs):
        super(TestLslRecorder, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        """Set up a mock LSL data server that is streaming for all tests."""
        cls.eeg_server = LslDataServer(device_spec=DEVICE)
        cls.eeg_server.start()

    @classmethod
    def tearDownClass(cls):
        """"Stop the data server."""
        cls.eeg_server.stop()

    def setUp(self):
        """Override; set up the needed path for load functions."""
        self.temp_dir = tempfile.mkdtemp()

    def test_recorder(self):
        """Test basic recording functionality"""
        path = Path(self.temp_dir, f'eeg_data_{DEVICE_NAME.lower()}.csv')
        recorder = LslRecorder(path=self.temp_dir)
        self.assertFalse(path.exists())
        recorder.start()
        time.sleep(0.1)
        self.assertTrue(path.exists())
        recorder.stop(wait=True)

        raw_data = load(path)
        self.assertEqual(raw_data.daq_type, DEVICE_NAME)
        self.assertEqual(raw_data.sample_rate, DEVICE.sample_rate)

        self.assertEqual(raw_data.columns[0], TIMESTAMP_COLUMN)
        self.assertEqual(raw_data.columns[1:-1], DEVICE.channels)
        self.assertEqual(raw_data.columns[-1], 'lsl_timestamp')
        self.assertTrue(len(raw_data.rows) > 0)

    def test_multiple_sources(self):
        """Test that recorder works with multiple sources and can be customized
        with filenames for each device type."""
        # create another server with a different device type
        server = eye_tracker_server()
        server.start()

        recorder = LslRecorder(path=self.temp_dir,
                               filenames={
                                   'EEG': 'raw_data_1.csv',
                                   'Gaze': 'gaze_data_1.csv'
                               })
        recorder.start()

        filenames = [stream.filename for stream in recorder.streams]
        self.assertEqual(2, len(filenames))

        time.sleep(0.1)
        recorder.stop(wait=True)
        server.stop()

        self.assertTrue('raw_data_1.csv' in filenames)
        self.assertTrue('gaze_data_1.csv' in filenames)
        self.assertTrue(Path(self.temp_dir, 'raw_data_1.csv').exists())
        self.assertTrue(Path(self.temp_dir, 'gaze_data_1.csv').exists())

    def test_duplicate_streams(self):
        """Test that an exception is thrown when there are multiple LSL streams
        with the same type and name."""
        dup_server = LslDataServer(device_spec=DEVICE)
        recorder = LslRecorder(path=self.temp_dir)

        dup_server.start()
        with pytest.raises(Exception):
            recorder.start()

        dup_server.stop()


if __name__ == '__main__':
    unittest.main()
