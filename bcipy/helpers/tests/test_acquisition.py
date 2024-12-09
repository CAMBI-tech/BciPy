"""Tests for acquisition helper."""
import unittest
from unittest.mock import Mock, call, patch

from bcipy.acquisition.devices import DeviceSpec, DeviceStatus
from bcipy.config import DEFAULT_DEVICE_SPEC_FILENAME as spec_name
from bcipy.helpers.acquisition import (RAW_DATA_FILENAME, StreamType,
                                       active_content_types, init_acquisition,
                                       init_device, is_stream_type_active,
                                       max_inquiry_duration, parse_stream_type,
                                       raw_data_filename, server_spec,
                                       stream_types)
from bcipy.io.load import load_json_parameters
from bcipy.io.save import init_save_data_structure


class TestAcquisition(unittest.TestCase):
    """Unit tests for acquisition helper main method"""

    @patch('bcipy.helpers.acquisition.save_device_specs')
    @patch('bcipy.helpers.acquisition.start_viewer')
    @patch('bcipy.helpers.acquisition.init_lsl_client')
    @patch('bcipy.helpers.acquisition.init_device')
    @patch('bcipy.helpers.acquisition.await_start')
    @patch('bcipy.helpers.acquisition.LslDataServer')
    @patch('bcipy.helpers.acquisition.server_spec')
    @patch('bcipy.helpers.acquisition.ClientManager')
    def test_init_acquisition(self, make_client_manager, server_spec_mock,
                              make_lsl_data_server, await_start_mock,
                              init_device_mock, init_lsl_client_mock,
                              start_viewer_mock, save_device_specs_mock):
        """Test init_acquisition with LSL client."""
        # Function parameters
        params = Parameters.from_cast_values(acq_mode='EEG:passive/DSI-24',
                                             acq_show_viewer=False)
        save_folder = "temp"

        # Mock objects
        manager = Mock()
        server_device = Mock()
        eeg_device = DeviceSpec(content_type='EEG',
                                name='DSI-24',
                                status=DeviceStatus.PASSIVE,
                                channels=['ch1', 'ch2', 'ch3'],
                                sample_rate=300.0)
        server = Mock()
        lsl_client = Mock()

        # Mock the functions/constructors
        make_client_manager.return_value = manager
        server_spec_mock.return_value = server_device
        make_lsl_data_server.return_value = server
        init_device_mock.return_value = eeg_device
        init_lsl_client_mock.return_value = lsl_client

        client_manager, servers = init_acquisition(params,
                                                   save_folder=save_folder,
                                                   server=True)

        # Assertions
        self.assertEqual(client_manager, manager)
        self.assertEqual(1, len(servers))
        self.assertEqual(server, servers[0])
        init_device_mock.assert_called_with('EEG', 'DSI-24',
                                            DeviceStatus.PASSIVE)
        await_start_mock.assert_called_once()
        start_viewer_mock.assert_not_called()
        manager.add_client.assert_called_with(lsl_client)
        save_device_specs_mock.assert_called_with(manager.device_specs,
                                                  save_folder, spec_name)

    @patch('bcipy.helpers.acquisition.save_device_specs')
    @patch('bcipy.helpers.acquisition.start_viewer')
    @patch('bcipy.helpers.acquisition.init_lsl_client')
    @patch('bcipy.helpers.acquisition.init_device')
    @patch('bcipy.helpers.acquisition.await_start')
    @patch('bcipy.helpers.acquisition.LslDataServer')
    @patch('bcipy.helpers.acquisition.server_spec')
    @patch('bcipy.helpers.acquisition.ClientManager')
    def test_init_acquisition_multiple_devices(
            self, make_client_manager, server_spec_mock, make_lsl_data_server,
            await_start_mock, init_device_mock, init_lsl_client_mock,
            start_viewer_mock, save_device_specs_mock):
        """Test init acquisition with multiple devices."""
        # Function parameters
        params = Parameters.from_cast_values(
            acq_mode='EEG/DSI-24+Eyetracker:passive',
            acq_show_viewer=True,
            stim_screen=0,
            parameter_location=".")
        save_folder = "temp"

        # Mock objects
        manager = Mock()
        # preconfigured devices
        eeg_device = DeviceSpec(content_type='EEG',
                                name='DSI-24',
                                status=DeviceStatus.ACTIVE,
                                channels=['ch1', 'ch2', 'ch3'],
                                sample_rate=300.0)
        gaze_device = DeviceSpec(content_type='Eyetracker',
                                 name='Tobii Nano',
                                 status=DeviceStatus.PASSIVE,
                                 channels=['left_pos', 'right_pos'],
                                 sample_rate=60)
        eeg_server = Mock()
        gaze_server = Mock()
        eeg_client = Mock()
        gaze_client = Mock()

        # Mock the functions/constructors
        make_client_manager.return_value = manager
        server_spec_mock.side_effect = [eeg_device, gaze_device]
        make_lsl_data_server.side_effect = [eeg_server, gaze_server]

        # get preconfigured device from devices.json
        init_device_mock.size_effect = [eeg_device, gaze_device]
        init_lsl_client_mock.side_effect = [eeg_client, gaze_client]

        client_manager, servers = init_acquisition(params,
                                                   save_folder=save_folder,
                                                   server=True)

        # Assertions
        self.assertEqual(client_manager, manager)
        self.assertEqual(2, len(servers))
        self.assertEqual(eeg_server, servers[0])
        self.assertEqual(gaze_server, servers[1])

        self.assertEqual(init_device_mock.call_args_list, [
            call('EEG', 'DSI-24', None),
            call('Eyetracker', None, DeviceStatus.PASSIVE)
        ])
        self.assertEqual(await_start_mock.call_count, 2)

        start_viewer_mock.assert_called_once()
        self.assertEqual(
            manager.add_client.call_args_list,
            [call(eeg_client), call(gaze_client)])
        save_device_specs_mock.assert_called_with(manager.device_specs,
                                                  save_folder, spec_name)


class TestAcquisitionHelpers(unittest.TestCase):
    """Unit tests for acquisition helper functions"""

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

    @patch('bcipy.helpers.acquisition.discover_device_spec')
    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_init_device_with_status_override(self, preconfigured_device_mock,
                                              discover_spec_mock):
        """Test device initialization where the provided status is different than
        the preconfigured status."""

        preconf_device = DeviceSpec(content_type='EEG',
                                    name='DSI-24',
                                    status=DeviceStatus.ACTIVE,
                                    channels=['ch1', 'ch2', 'ch3'],
                                    sample_rate=300.0)

        preconfigured_device_mock.return_value = preconf_device

        device_spec = init_device('EEG', 'DSI-24', DeviceStatus.PASSIVE)

        discover_spec_mock.assert_not_called()
        preconfigured_device_mock.assert_called_with('DSI-24', strict=True)

        self.assertEqual(device_spec.name, preconf_device.name)
        self.assertEqual(device_spec.content_type, preconf_device.content_type)
        self.assertEqual(device_spec.status, DeviceStatus.PASSIVE)

    @patch('bcipy.helpers.acquisition.discover_device_spec')
    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_init_device_with_active_status_override(self,
                                                     preconfigured_device_mock,
                                                     discover_spec_mock):
        """Test device initialization where the provided active status is
        different than the preconfigured status."""

        preconf_device = DeviceSpec(content_type='EEG',
                                    name='DSI-24',
                                    status=DeviceStatus.PASSIVE,
                                    channels=['ch1', 'ch2', 'ch3'],
                                    sample_rate=300.0)
        preconfigured_device_mock.return_value = preconf_device

        device_spec = init_device('EEG', 'DSI-24', DeviceStatus.ACTIVE)
        self.assertEqual(device_spec.status, DeviceStatus.ACTIVE)
        discover_spec_mock.assert_not_called()

    def test_parse_stream_type(self):
        """Test function to split the stream type into content_type, name,
        and status"""
        self.assertEqual(('EEG', 'DSI-24', None),
                         parse_stream_type('EEG/DSI-24'))
        self.assertEqual(('Gaze', None, None), parse_stream_type('Gaze'))
        self.assertEqual(('Gaze', None, DeviceStatus.PASSIVE),
                         parse_stream_type('Gaze:passive'))
        self.assertEqual(('EEG', 'DSI-24', DeviceStatus.ACTIVE),
                         parse_stream_type('EEG:active/DSI-24'))

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

    def test_raw_data_filename_eeg(self):
        """Test generation of filename for EEG devices"""
        device = DeviceSpec(name='DSI-24',
                            channels=[],
                            sample_rate=300.0,
                            content_type='EEG')
        self.assertEqual(raw_data_filename(device), f'{RAW_DATA_FILENAME}.csv')

    def test_raw_data_filename_eye_tracker(self):
        """Test generation of filename for EyeTracker devices"""

        device = DeviceSpec(name='Tobii-P0',
                            channels=[],
                            sample_rate=60,
                            content_type='EYETRACKER')
        self.assertEqual(raw_data_filename(device),
                         'eyetracker_data_tobii-p0.csv')

    def test_stream_type_active_given_status(self):
        """Test function to test if a StreamType is active given the provided
        status."""

        self.assertTrue(
            is_stream_type_active(
                StreamType(content_type='EEG', status=DeviceStatus.ACTIVE)))

        self.assertFalse(
            is_stream_type_active(
                StreamType(content_type='EEG', status=DeviceStatus.PASSIVE)))

    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_stream_type_active_using_preconfigured(self,
                                                    preconfigured_device_mock):
        """Test function to test if a StreamType is active using a
        preconfigured device."""

        stream_type = StreamType(content_type='EEG', device_name='MyDevice')
        device1 = Mock()
        device1.is_active = True
        preconfigured_device_mock.return_value = device1

        self.assertTrue(is_stream_type_active(stream_type))
        preconfigured_device_mock.assert_called_with('MyDevice', strict=False)

        device1.is_active = False
        self.assertFalse(is_stream_type_active(stream_type))

    @patch('bcipy.helpers.acquisition.preconfigured_device')
    @patch('bcipy.helpers.acquisition.with_content_type')
    def test_stream_type_active_using_content_type(self,
                                                   with_content_type_mock,
                                                   preconfigured_device_mock):
        """Test function to test if a StreamType is active using a
        preconfigured device with type."""

        stream_type = StreamType(content_type='EEG', device_name='MyDevice')

        preconfigured_device_mock.return_value = None
        device1 = Mock()
        device1.is_active = False
        device2 = Mock()
        device2.is_active = False
        with_content_type_mock.return_value = [device1, device2]

        self.assertFalse(is_stream_type_active(stream_type))
        with_content_type_mock.assert_called_with('EEG')

        device1.is_active = True
        self.assertTrue(is_stream_type_active(stream_type))

    def test_active_content_types_with_declared_spec(self):
        """Test active content types when the spec declares the status."""
        self.assertListEqual([], active_content_types('EEG:passive'))
        self.assertListEqual(
            [], active_content_types('EEG:passive+Eyetracker:passive'))
        self.assertListEqual(
            ['EEG'], active_content_types('EEG:active+Eyetracker:passive'))
        self.assertListEqual(
            ['EEG', 'Eyetracker'],
            active_content_types('EEG:active+Eyetracker:active'))
        self.assertListEqual(
            ['Eyetracker'],
            active_content_types('EEG:passive+Eyetracker:active'))

    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_active_content_types_using_preconfigured(
            self, preconfigured_device_mock):
        """Test active content types using a preconfigured device."""

        device1 = Mock()
        device1.is_active = True
        preconfigured_device_mock.return_value = device1

        self.assertListEqual(['EEG'], active_content_types('EEG/MyDevice'))
        preconfigured_device_mock.assert_called_with('MyDevice', strict=False)

    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_active_content_types_using_preconfigured_false(
            self, preconfigured_device_mock):
        """Test active content types using a preconfigured device."""

        device1 = Mock()
        device1.is_active = False
        preconfigured_device_mock.return_value = device1

        self.assertListEqual([], active_content_types('EEG/MyDevice'))
        preconfigured_device_mock.assert_called_with('MyDevice', strict=False)

    @patch('bcipy.helpers.acquisition.preconfigured_device')
    def test_active_content_types_using_preconfigured_multiple_devices(
            self, preconfigured_device_mock):
        """Test active content types using a preconfigured device."""

        device1 = Mock()
        device1.is_active = False
        device2 = Mock()
        device2.is_active = True
        preconfigured_device_mock.side_effect = [device1, device2]

        self.assertListEqual(
            ['Eyetracker'],
            active_content_types('EEG/MyDevice+Eyetracker/MyEyeDevice'))

    @patch('bcipy.helpers.acquisition.with_content_type')
    def test_active_content_types_using_content_type_false(
            self, with_content_type_mock):
        """Test active content types using content type."""

        device1 = Mock()
        device1.is_active = False
        device2 = Mock()
        device2.is_active = False
        with_content_type_mock.return_value = [device1, device2]

        self.assertListEqual([], active_content_types('EEG'))
        with_content_type_mock.assert_called_with('EEG')

    @patch('bcipy.helpers.acquisition.with_content_type')
    def test_active_content_types_using_content_type(self,
                                                     with_content_type_mock):
        """Test active content types using content type."""
        device1 = Mock()
        device1.is_active = False
        device2 = Mock()
        device2.is_active = True
        with_content_type_mock.return_value = [device1, device2]

        self.assertListEqual(['EEG'], active_content_types('EEG'))
        with_content_type_mock.assert_called_with('EEG')


if __name__ == '__main__':
    unittest.main()
