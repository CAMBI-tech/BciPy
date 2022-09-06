"""Tests for hardware device specification."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from bcipy.acquisition import devices
from bcipy.acquisition.connection_method import ConnectionMethod

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
        self.assertTrue(len(supported.keys()) > 0)
        self.assertTrue('LSL' in supported.keys())
        self.assertTrue('DSI' in supported.keys())

        dsi = supported['DSI']
        self.assertEqual('EEG', dsi.content_type)

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
                 connection_methods=["TCP", "LSL"],
                 description="Wearable Sensing DSI-VR300")
        ]
        config_path = Path(temp_dir, 'my_devices.json')
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as config_file:
            json.dump(my_devices, config_file)

        devices.load(config_path)
        supported = devices.preconfigured_devices()
        self.assertEqual(1, len(supported.keys()))
        self.assertTrue('DSI-VR300' in supported.keys())

        spec = supported['DSI-VR300']
        self.assertEqual('EEG', spec.content_type)
        self.assertEqual(300.0, spec.sample_rate)
        self.assertEqual(channels, spec.channels)

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
                 connection_methods=["TCP", "LSL"],
                 description="My custom device")
        ]
        config_path = Path(temp_dir, 'my_devices.json')
        with open(config_path, 'w', encoding=DEFAULT_ENCODING) as config_file:
            json.dump(my_devices, config_file)

        devices.load(config_path)
        supported = devices.preconfigured_devices()

        spec = supported["Custom-Device"]
        self.assertEqual(spec.channels, ['Fz', 'Pz', 'F7'])
        self.assertEqual(spec.channel_names, ['ch1', 'ch2', 'ch3'])

        shutil.rmtree(temp_dir)

    def test_device_registration(self):
        """Should be able to register a new device spec"""

        data = dict(
            name="my-device",
            content_type="EEG",
            channels=["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz", "TRG"],
            sample_rate=300.0,
            connection_methods=["TCP", "LSL"],
            description="Custom built device")
        supported = devices.preconfigured_devices()
        device_count = len(supported.keys())
        self.assertTrue(device_count > 0)
        self.assertTrue('my-device' not in supported.keys())

        spec = devices.make_device_spec(data)
        devices.register(spec)

        supported = devices.preconfigured_devices()
        self.assertEqual(device_count + 1, len(supported.keys()))
        self.assertTrue('my-device' in supported.keys())

    def test_search_by_name(self):
        """Should be able to find a supported device by name."""
        dsi_device = devices.preconfigured_device('DSI')
        self.assertEqual('DSI', dsi_device.name)

    def test_device_spec_defaults(self):
        """DeviceSpec should require minimal information with default values."""
        spec = devices.DeviceSpec(name='TestDevice',
                                  channels=['C1', 'C2', 'C3'],
                                  sample_rate=256.0)
        self.assertTrue(ConnectionMethod.LSL in spec.connection_methods)
        self.assertEqual(3, spec.channel_count)
        self.assertEqual('EEG', spec.content_type)

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
