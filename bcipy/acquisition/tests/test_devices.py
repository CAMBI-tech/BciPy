"""Tests for hardware device specification."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from bcipy.acquisition import devices
from bcipy.config import DEFAULT_ENCODING


class TestDeviceSpecs(unittest.TestCase):
    """Tests for device specification."""

    def setUp(self):
        """Reload default devices"""
        devices.load()

    def test_default_supported_devices(self):
        """List of supported devices should include generic values for
        backwards compatibility."""
        supported = devices.preconfigured_devices()
        self.assertTrue(len(supported) > 0)
        self.assertTrue('DSI-24' in supported)

        dsi = supported['DSI-24']
        self.assertEqual('EEG', dsi.content_type)

        self.assertEqual(len(devices.with_content_type('EEG')), 4)

    def test_load_from_config(self):
        """Should be able to load a list of supported devices from a
        configuration file."""

        # create a config file in a temp location.
        temp_dir = tempfile.mkdtemp()
        channels = ["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"]
        my_devices = [
            dict(name="DSI-VR300",
                 content_type="EEG",
                 channels=channels,
                 sample_rate=300.0,
                 description="Wearable Sensing DSI-VR300")
        ]
        config_path = Path(temp_dir, 'my_devices.json')
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as config_file:
            json.dump(my_devices, config_file)

        devices.load(config_path, replace=True)
        supported = devices.preconfigured_devices()
        self.assertEqual(1, len(supported))
        self.assertTrue('DSI-VR300' in supported)

        spec = supported['DSI-VR300']
        self.assertEqual('EEG', spec.content_type)
        self.assertEqual(300.0, spec.sample_rate)
        self.assertEqual(channels, spec.channels)
        self.assertEqual(devices.DeviceStatus.ACTIVE, spec.status)
        self.assertTrue(spec.is_active)

        self.assertEqual(spec, devices.preconfigured_device('DSI-VR300'))
        shutil.rmtree(temp_dir)

    def test_load_channel_specs_from_config(self):
        """Channel specs should be loaded correctly from a configuration file."""

        # create a config file in a temp location.
        temp_dir = tempfile.mkdtemp()
        my_devices = [
            dict(name="Custom-Device",
                 content_type="EEG",
                 channels=[{
                     "name": "ch1",
                     "label": "Fz",
                     "units": "microvolts",
                     "type": "EEG"
                 }, {
                     "name": "ch2",
                     "label": "Pz",
                     "units": "microvolts",
                     "type": "EEG"
                 }, {
                     "name": "ch3",
                     "label": "F7"
                 }],
                 sample_rate=300.0,
                 description="My custom device")
        ]
        config_path = Path(temp_dir, 'my_devices.json')
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as config_file:
            json.dump(my_devices, config_file)

        prior_device_count = len(devices.preconfigured_devices())
        devices.load(config_path)
        supported = devices.preconfigured_devices()

        spec = supported["Custom-Device"]
        self.assertEqual(spec.channels, ['Fz', 'Pz', 'F7'])
        self.assertEqual(spec.channel_names, ['ch1', 'ch2', 'ch3'])
        self.assertEqual(len(supported), prior_device_count + 1)
        shutil.rmtree(temp_dir)

    def test_load_missing_config(self):
        """Missing config should not error and should not overwrite."""

        # create a config file in a temp location.
        temp_dir = tempfile.mkdtemp()
        devices.load(config_path=Path(temp_dir, 'does_not_exist.json'),
                     replace=True)
        self.assertTrue(devices.preconfigured_devices(),
                        msg="Default devices should still be configured")
        shutil.rmtree(temp_dir)

    def test_device_registration(self):
        """Should be able to register a new device spec"""

        data = dict(
            name="my-device",
            content_type="EEG",
            channels=["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"],
            sample_rate=300.0,
            description="Custom built device")
        supported = devices.preconfigured_devices()
        device_count = len(supported)
        self.assertTrue(device_count > 0)
        self.assertTrue('my-device' not in supported)

        spec = devices.make_device_spec(data)
        devices.register(spec)

        supported = devices.preconfigured_devices()
        self.assertEqual(device_count + 1, len(supported))
        self.assertTrue('my-device' in supported)

    def test_search_by_name(self):
        """Should be able to find a supported device by name."""
        dsi_device = devices.preconfigured_device('DSI-24')
        self.assertEqual('DSI-24', dsi_device.name)

    def test_device_spec_defaults(self):
        """DeviceSpec should require minimal information with default values."""
        spec = devices.DeviceSpec(name='TestDevice',
                                  channels=['C1', 'C2', 'C3'],
                                  sample_rate=256.0)
        self.assertEqual(3, spec.channel_count)
        self.assertEqual('EEG', spec.content_type)
        self.assertEqual(devices.DeviceStatus.ACTIVE, spec.status)

    def test_device_spec_analysis_channels(self):
        """DeviceSpec should have a list of channels used for analysis."""
        spec = devices.DeviceSpec(name='TestDevice',
                                  channels=['C1', 'C2', 'C3', 'TRG'],
                                  sample_rate=256.0,
                                  excluded_from_analysis=['TRG'])

        self.assertEqual(['C1', 'C2', 'C3'], spec.analysis_channels)
        spec.excluded_from_analysis = []
        self.assertEqual(['C1', 'C2', 'C3', 'TRG'], spec.analysis_channels)

        spec2 = devices.DeviceSpec(name='Device2',
                                   channels=['C1', 'C2', 'C3', 'TRG'],
                                   sample_rate=256.0,
                                   excluded_from_analysis=['C1', 'TRG'])
        self.assertEqual(['C2', 'C3'], spec2.analysis_channels)

        spec3 = devices.DeviceSpec(name='Device3',
                                   channels=[{
                                       'name': 'C1',
                                       'label': 'ch1'
                                   }, {
                                       'name': 'C2',
                                       'label': 'ch2'
                                   }, {
                                       'name': 'C3',
                                       'label': 'ch3'
                                   }, {
                                       'name': 'C4',
                                       'label': 'TRG'
                                   }],
                                   sample_rate=256.0,
                                   excluded_from_analysis=['ch1', 'TRG'])
        self.assertEqual(['ch2', 'ch3'], spec3.analysis_channels)

    def test_irregular_sample_rate(self):
        """Test that DeviceSpec supports an IRREGULAR sample rate."""
        spec = devices.DeviceSpec(name='Mouse',
                                  channels=['Btn1', 'Btn2'],
                                  sample_rate=devices.IRREGULAR_RATE,
                                  content_type='Markers')
        self.assertEqual(devices.IRREGULAR_RATE, spec.sample_rate)
        with self.assertRaises(AssertionError):
            devices.DeviceSpec(name='Mouse',
                               channels=['Btn1', 'Btn2'],
                               sample_rate=-100.0,
                               content_type='Markers')

    def test_data_type(self):
        """Test that DeviceSpec supports a data type."""
        spec = devices.DeviceSpec(name='Mouse',
                                  channels=['Btn1', 'Btn2'],
                                  sample_rate=devices.IRREGULAR_RATE,
                                  content_type='Markers')
        self.assertEqual('float32', spec.data_type,
                         "Should have a default type")

        spec = devices.DeviceSpec(name='Mouse',
                                  channels=['Btn1', 'Btn2'],
                                  sample_rate=devices.IRREGULAR_RATE,
                                  content_type='Markers',
                                  data_type='string')
        self.assertEqual('string', spec.data_type)

        # Should raise an exception for an invalid data type.
        with self.assertRaises(AssertionError):
            devices.DeviceSpec(name='Mouse',
                               channels=['Btn1', 'Btn2'],
                               sample_rate=devices.IRREGULAR_RATE,
                               content_type='Markers',
                               data_type='Whatever')

    def test_device_spec_to_dict(self):
        """DeviceSpec should be able to be converted to a dictionary."""
        device_name = 'TestDevice'
        channels = ['C1', 'C2', 'C3']
        expected_channel_output = [{'label': 'C1', 'name': 'C1', 'type': None, 'units': None},
                                   {'label': 'C2', 'name': 'C2', 'type': None, 'units': None},
                                   {'label': 'C3', 'name': 'C3', 'type': None, 'units': None}]
        sample_rate = 256.0
        content_type = 'EEG'
        spec = devices.DeviceSpec(name=device_name,
                                  channels=channels,
                                  sample_rate=sample_rate,
                                  content_type=content_type,
                                  status=devices.DeviceStatus.PASSIVE)
        spec_dict = spec.to_dict()
        self.assertEqual(device_name, spec_dict['name'])
        self.assertEqual(content_type, spec_dict['content_type'])
        self.assertEqual(expected_channel_output, spec_dict['channels'])
        self.assertEqual(sample_rate, spec_dict['sample_rate'])
        self.assertEqual('passive', spec_dict['status'])

    def test_load_status(self):
        """Should be able to load a list of supported devices from a
        configuration file."""

        # create a config file in a temp location.
        temp_dir = tempfile.mkdtemp()
        my_devices = [
            dict(name="MyDevice",
                 content_type="EEG",
                 description="My Device",
                 channels=["a", "b", "c"],
                 sample_rate=100.0,
                 status=str(devices.DeviceStatus.PASSIVE))
        ]
        config_path = Path(temp_dir, 'my_devices.json')
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as config_file:
            json.dump(my_devices, config_file)

        devices.load(config_path, replace=True)
        supported = devices.preconfigured_devices()
        self.assertEqual(devices.DeviceStatus.PASSIVE, supported['MyDevice'].status)
        shutil.rmtree(temp_dir)

    def test_device_status(self):
        """Test DeviceStatus enum"""
        self.assertEqual('active', str(devices.DeviceStatus.ACTIVE))
        self.assertEqual(devices.DeviceStatus.ACTIVE,
                         devices.DeviceStatus.from_str('active'))
        self.assertEqual(
            devices.DeviceStatus.PASSIVE,
            devices.DeviceStatus.from_str(str(devices.DeviceStatus.PASSIVE)))


if __name__ == '__main__':
    unittest.main()
