import unittest
from unittest.mock import MagicMock, Mock

from bcipy.acquisition.exceptions import InsufficientDataException
from bcipy.acquisition.multimodal import ClientManager, ContentType


class TestClientManager(unittest.TestCase):
    """Tests for multimodal client manager."""

    def setUp(self):
        """Setup common state"""
        self.eeg_device_mock = Mock()
        self.eeg_device_mock.name = 'DSI-24'
        self.eeg_device_mock.content_type = 'EEG'
        self.eeg_device_mock.sample_rate = 300
        self.eeg_device_mock.is_active = True

        self.eeg_client_mock = Mock()
        self.eeg_client_mock.device_spec = self.eeg_device_mock
        self.eeg_data_mock = Mock()
        self.eeg_client_mock.get_data = MagicMock(
            return_value=self.eeg_data_mock)

        self.gaze_device_mock = Mock()
        self.gaze_device_mock.content_type = 'Eyetracker'
        self.gaze_device_mock.sample_rate = 60
        self.gaze_device_mock.is_active = False
        self.gaze_client_mock = Mock()
        self.gaze_client_mock.device_spec = self.gaze_device_mock

    def test_add_client(self):
        """Test adding a client"""
        manager = ClientManager()
        self.assertEqual(manager.get_client(ContentType.EEG), None)
        manager.add_client(self.eeg_client_mock)
        self.assertEqual(manager.get_client(ContentType.EEG),
                         self.eeg_client_mock)

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

        self.assertEqual(manager.default_content_type, ContentType.EEG)
        self.assertEqual(manager.default_client, self.eeg_client_mock)

        manager.default_content_type = ContentType.EYETRACKER
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

    def test_device_content_types(self):
        """Test properties related to content types"""
        manager = ClientManager()
        manager.add_client(self.eeg_client_mock)
        manager.add_client(self.gaze_client_mock)
        self.assertTrue(ContentType.EEG in manager.device_content_types)
        self.assertTrue(ContentType.EYETRACKER in manager.device_content_types)

        self.assertEqual([ContentType.EEG],
                         manager.active_device_content_types)

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

        self.eeg_client_mock.get_data.assert_called_once_with(start=100,
                                                              limit=1000)

    def test_get_data_by_device(self):
        """Test getting data for multiple devices"""
        # Set up switch device mock
        switch_device_mock = Mock()
        switch_device_mock.name = 'Test-switch-2000'
        switch_device_mock.content_type = 'Markers'
        switch_device_mock.sample_rate = 0.0

        switch_client_mock = Mock()
        switch_client_mock.device_spec = switch_device_mock
        switch_data_mock = Mock()
        switch_data_mock.__len__ = lambda self: 5
        switch_client_mock.get_data = MagicMock(return_value=switch_data_mock)

        gaze_data_mock = Mock()
        gaze_data_mock.__len__ = lambda self: 300
        self.gaze_client_mock.get_data = MagicMock(return_value=gaze_data_mock)

        self.eeg_data_mock.__len__ = lambda self: 1500

        daq = ClientManager()
        daq.add_client(self.eeg_client_mock)
        daq.add_client(self.gaze_client_mock)
        daq.add_client(switch_client_mock)

        results = daq.get_data_by_device(start=100,
                                         seconds=5,
                                         content_types=[
                                             ContentType.EEG,
                                             ContentType.EYETRACKER,
                                             ContentType.MARKERS
                                         ])

        self.eeg_client_mock.get_data.assert_called_once_with(start=100,
                                                              limit=1500)
        self.gaze_client_mock.get_data.assert_called_once_with(start=100,
                                                               limit=300)
        switch_client_mock.get_data.assert_called_with(start=100, end=105)

        self.assertTrue(ContentType.EEG in results)
        self.assertTrue(ContentType.EYETRACKER in results)
        self.assertTrue(ContentType.MARKERS in results)

    def test_insufficient_data(self):
        """Test insufficient data exception in strict mode."""

        gaze_data_mock = Mock()
        gaze_data_mock.__len__ = lambda self: 299
        self.gaze_client_mock.get_data = MagicMock(return_value=gaze_data_mock)

        daq = ClientManager()
        daq.add_client(self.gaze_client_mock)

        with self.assertRaises(InsufficientDataException):
            daq.get_data_by_device(start=100, seconds=5)

    def test_non_strict_data_request(self):
        """Test data request in non-strict mode."""

        gaze_data_mock = Mock()
        gaze_data_mock.__len__ = lambda self: 299
        self.gaze_client_mock.get_data = MagicMock(return_value=gaze_data_mock)

        daq = ClientManager()
        daq.add_client(self.gaze_client_mock)

        results = daq.get_data_by_device(start=100, seconds=5, strict=False)
        self.gaze_client_mock.get_data.assert_called_once_with(start=100,
                                                               limit=300)
        self.assertTrue(ContentType.EYETRACKER in results)


if __name__ == '__main__':
    unittest.main()
