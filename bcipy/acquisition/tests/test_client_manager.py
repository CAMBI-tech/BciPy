import unittest
from unittest.mock import Mock
from bcipy.acquisition.multimodal import ClientManager


class TestClientManager(unittest.TestCase):
    """Tests for multimodal client manager."""

    def setUp(self):
        """Setup common state"""
        self.eeg_device_mock = Mock()
        self.eeg_device_mock.name = 'DSI-24'
        self.eeg_device_mock.content_type = 'EEG'

        self.eeg_client_mock = Mock()
        self.eeg_client_mock.device_spec = self.eeg_device_mock
        self.eeg_client_mock.get_data = Mock()

        self.gaze_device_mock = Mock()
        self.gaze_device_mock.content_type = 'Eyetracker'
        self.gaze_client_mock = Mock()
        self.gaze_client_mock.device_spec = self.gaze_device_mock

    def test_add_client(self):
        """Test adding a client"""
        manager = ClientManager()
        self.assertEqual(manager.get_client('EEG'), None)
        manager.add_client(self.eeg_client_mock)
        self.assertEqual(manager.get_client('EEG'), self.eeg_client_mock)

    def test_start_acquisition(self):
        """Test that manager can start acquisition in clients"""
        manager = ClientManager()
        manager.add_client(self.eeg_client_mock)
        manager.add_client(self.gaze_client_mock)

        manager.start_acquisition()
        self.eeg_client_mock.start_acquisition.assert_called_once()
        self.gaze_client_mock.start_acquisition.assert_called_once()

    def test_stop_acquisition(self):
        """Test that manager can stop acquisition in clients"""
        manager = ClientManager()
        manager.add_client(self.eeg_client_mock)
        manager.add_client(self.gaze_client_mock)

        manager.start_acquisition()
        manager.stop_acquisition()
        self.eeg_client_mock.stop_acquisition.assert_called_once()
        self.gaze_client_mock.stop_acquisition.assert_called_once()

    def test_default_client(self):
        """Test default client property."""
        manager = ClientManager()
        manager.add_client(self.eeg_client_mock)
        manager.add_client(self.gaze_client_mock)

        self.assertEqual(manager.default_client, self.eeg_client_mock)

        manager.default_content_type = self.gaze_device_mock.content_type
        self.assertEqual(manager.default_client, self.gaze_client_mock)

    def test_device_specs(self):
        """Test the device_specs property"""
        manager = ClientManager()
        manager.add_client(self.eeg_client_mock)

        self.assertEqual(1, len(manager.device_specs))
        self.assertTrue(self.eeg_device_mock in manager.device_specs)

        manager.add_client(self.gaze_client_mock)
        self.assertEqual(2, len(manager.device_specs))
        self.assertTrue(self.gaze_device_mock in manager.device_specs)

    def test_dispatching_properties(self):
        """Test that property calls may be dispatched to the default client"""
        manager = ClientManager()
        manager.add_client(self.eeg_client_mock)
        self.assertEqual(manager.device_spec, self.eeg_device_mock)

    def test_dispatching_methods(self):
        """Test that method calls may be dispatched to the default client"""
        daq = ClientManager()
        daq.add_client(self.eeg_client_mock)
        daq.get_data(start=100, limit=1000)

        self.eeg_client_mock.get_data.assert_called_once_with(start=100, limit=1000)


if __name__ == '__main__':
    unittest.main()
