"""Tests for acquisition helper."""
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.helpers.acquisition import (init_device, init_eeg_acquisition,
                                       max_inquiry_duration, parse_stream_type,
                                       server_spec, stream_types)
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.save import init_save_data_structure


class TestAcquisition(unittest.TestCase):
    """Unit tests for acquisition helper"""

    def setUp(self):
        """set up the needed path for load functions."""
        self.parameters_used = DEFAULT_PARAMETERS_PATH
        self.parameters = load_json_parameters(self.parameters_used,
                                               value_cast=True)
        self.data_save_path = 'data/'
        self.user_information = 'test_user_001'
        self.task = 'RSVP Calibration'

        self.save = init_save_data_structure(self.data_save_path,
                                             self.user_information,
                                             self.parameters_used, self.task)

    def tearDown(self):
        """Override; teardown test"""
        shutil.rmtree(self.save)

    def test_init_acquisition(self):
        """Test init_eeg_acquisition with LSL client."""

        params = self.parameters
        params['acq_mode'] = 'EEG/DSI-24'

        client, servers = init_eeg_acquisition(params, self.save, server=True)

        client.stop_acquisition()
        client.cleanup()
        for server in servers:
            server.stop()

        self.assertEqual(1, len(servers))
        self.assertEqual(client.device_spec.name, 'DSI-24')

        self.assertTrue(Path(self.save, 'devices.json').is_file())

    def test_max_inquiry_duration(self):
        """Test the max inquiry duration function"""
        params = {
            'time_fixation': 0.5,
            'time_prompt': 1,
            'stim_length': 10,
            'time_flash': 0.25,
            'task_buffer_length': 0.75,
            'prestim_length': 0,
            'stim_jitter': 0
        }

        self.assertEqual(4.75, max_inquiry_duration(params))

    def test_parse_stream_types(self):
        """Test parsing the acq_mode parameter"""
        self.assertListEqual(['EEG', 'Gaze'], stream_types('EEG+Gaze'))
        self.assertListEqual(['EEG', 'Gaze'], stream_types(' EEG+Gaze'))
        self.assertListEqual(['EEG', 'Gaze'], stream_types('EEG + Gaze'))
        self.assertListEqual(['EEG', 'My stream'],
                             stream_types('EEG + My stream'),
                             "Should not validate the type itself.")

        self.assertListEqual(['EEG'], stream_types('EEG'))
        self.assertListEqual(['Gaze', 'EEG', 'EOG'],
                             stream_types('Gaze+EEG+EOG'))

        self.assertListEqual(['EEG', 'Gaze+'],
                             stream_types('EEG|Gaze+', delimiter='|'))

        self.assertListEqual(['EEG'], stream_types('EEG+EEG'))

    @patch('bcipy.helpers.acquisition.discover_device_spec')
    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_init_device_using_discovery(self, preconfigured_device_mock,
                                         discover_spec_mock):
        """Test device initialization where spec is derived from the LSL
        stream."""
        device_name = 'TestDevice'
        device_type = 'EEG'

        device_mock = Mock()
        device_mock.name = device_name
        discover_spec_mock.return_value = device_mock
        preconfigured_device_mock.return_value = None

        device_spec = init_device(device_type)

        discover_spec_mock.assert_called_with(device_type)
        preconfigured_device_mock.assert_called_with(device_name, strict=False)
        self.assertEqual(device_spec, device_mock)

    @patch('bcipy.helpers.acquisition.discover_device_spec')
    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_init_device_using_preconfigured(self, preconfigured_device_mock,
                                             discover_spec_mock):
        """Test device initialization where the discovered device_spec is
        overridden using a preconfigured device."""
        device_name = 'TestDevice'
        device_type = 'EEG'

        discovery_device_mock = Mock()
        discovery_device_mock.name = device_name

        device_mock = Mock()

        discover_spec_mock.return_value = discovery_device_mock
        preconfigured_device_mock.return_value = device_mock

        device_spec = init_device(device_type)

        discover_spec_mock.assert_called_with(device_type)
        preconfigured_device_mock.assert_called_with(device_name, strict=False)
        self.assertEqual(device_spec, device_mock)

    @patch('bcipy.helpers.acquisition.discover_device_spec')
    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_init_device_with_named_device(self, preconfigured_device_mock,
                                           discover_spec_mock):
        """Test device initialization where preconfigured device name is
        is provided."""

        device_mock = Mock()
        preconfigured_device_mock.return_value = device_mock

        device_spec = init_device('EEG', 'DSI-24')

        discover_spec_mock.assert_not_called()
        preconfigured_device_mock.assert_called_with('DSI-24', strict=True)
        self.assertEqual(device_spec, device_mock)

    def test_parse_stream_type(self):
        """Test function to split the stream type into content_type, name"""
        self.assertEqual(('EEG', 'DSI-24'), parse_stream_type('EEG/DSI-24'))
        self.assertEqual(('Gaze', None), parse_stream_type('Gaze'))

    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_server_spec_with_named_device(self, preconfigured_device_mock):
        """Test function to get the DeviceSpec for a mock server."""
        device_mock = Mock()
        preconfigured_device_mock.return_value = device_mock
        device_spec = server_spec(content_type='EEG', device_name='TestDevice')
        preconfigured_device_mock.assert_called_with('TestDevice', strict=True)
        self.assertEqual(device_spec, device_mock)

    @patch('bcipy.helpers.acquisition.with_content_type')
    def test_server_spec_without_named_device(self, with_content_type_mock):
        """Test function to get the DeviceSpec for a mock server."""
        device1 = Mock()
        device2 = Mock()
        with_content_type_mock.return_value = [device1, device2]

        device_spec = server_spec(content_type='EEG')
        with_content_type_mock.assert_called_with('EEG')
        self.assertEqual(device_spec, device1)


if __name__ == '__main__':
    unittest.main()
