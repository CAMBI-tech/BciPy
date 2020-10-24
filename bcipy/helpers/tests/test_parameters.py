"""Tests for Parameter related functionality"""
import shutil
import tempfile
import unittest
from collections import abc
from pathlib import Path

from bcipy.helpers.parameters import Parameters


class TestParameters(unittest.TestCase):
    """Tests for loading and saving Parameters."""

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters_location = 'bcipy/parameters/parameters.json'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_json_parameters_returns_dict(self):
        """Test load parameters returns a Python dict."""

        parameters = Parameters(source=self.parameters_location)
        self.assertTrue(isinstance(parameters, abc.MutableMapping))

    def test_load_json_parameters_throws_error_on_wrong_path(self):
        """Test load parameters returns error on entering wrong path."""

        with self.assertRaises(Exception):
            Parameters(source='/garbage/dir/wont/work')

    def test_load_initializes_parameters(self):
        """Test load parameters initializes data"""
        parameters = Parameters(source=self.parameters_location)
        self.assertTrue(len(parameters) > 0)

    def test_initialization_without_source(self):
        """Test parameter initialization without a JSON source file"""

        parameters = Parameters(source=None)
        self.assertTrue(isinstance(parameters, abc.MutableMapping))
        self.assertEqual(len(parameters), 0)

    def test_load_data(self):
        """Test load data"""
        sample_config = {
            "fake_data": {
                "value": "true",
                "section": "bci_config",
                "readableName": "Fake EEG Data On/Off",
                "helpTip":
                "If ‘true’, fake EEG data will be used instead of real EEG data.",
                "recommended_values": "",
                "type": "bool"
            },
            "acq_device": {
                "value": "LSL",
                "section": "acq_config",
                "readableName": "Acquisition Device Connection Method",
                "helpTip":
                "Specifies the method used to connect to the data acquisition device (LSL or DSI).",
                "recommended_values": ["DSI", "LSL"],
                "type": "str"
            }
        }
        parameters = Parameters(source=None)
        parameters.load(sample_config)
        self.assertEqual(len(parameters), 2)

    def test_load_data_with_missing_field(self):
        """Test that an exception is raised if data has a missing field."""
        data_with_missing_help_tip = {
            "fake_data": {
                "value": "true",
                "section": "bci_config",
                "readableName": "Fake EEG Data On/Off",
                "recommended_values": "",
                "type": "bool"
            }
        }
        parameters = Parameters(source=None)
        with self.assertRaises(Exception):
            parameters.load(data_with_missing_help_tip)

    def test_load_data_with_unsupported_type(self):
        """Test that an exception is raised if data has an unsupported data type."""
        data_with_unsupported_type = {
            "fake_data": {
                "value": "true",
                "section": "bci_config",
                "readableName": "Fake EEG Data On/Off",
                "helpTip": "",
                "recommended_values": "",
                "type": "custom_type"
            }
        }
        parameters = Parameters(source=None)
        with self.assertRaises(Exception):
            parameters.load(data_with_unsupported_type)

    def test_cast_values(self):
        """Test cast_values"""

        sample_config = {
            "myint": {
                "value": "1",
                "section": "",
                "readableName": "",
                "helpTip": "",
                "recommended_values": "",
                "type": "int"
            },
            "mybool": {
                "value": "true",
                "section": "",
                "readableName": "",
                "helpTip": "",
                "recommended_values": "",
                "type": "bool"
            },
            "mypath": {
                "value": "bcipy/parameters/parameters.json",
                "section": "",
                "readableName": "",
                "helpTip": "",
                "recommended_values": "",
                "type": "directorypath"
            },
            "mystr": {
                "value": "hello",
                "section": "",
                "readableName": "",
                "helpTip": "",
                "recommended_values": "",
                "type": "str"
            },
        }
        parameters = Parameters(source=None, cast_values=True)
        parameters.load(sample_config)

        self.assertEqual(parameters['myint'], 1)
        self.assertEqual(parameters.get('myint', 0), 1)
        self.assertEqual(parameters['mybool'], True)
        self.assertEqual(parameters.get('mybool', False), True)
        self.assertEqual(parameters['mypath'],
                         'bcipy/parameters/parameters.json')
        self.assertEqual(parameters['mystr'], 'hello')
        self.assertFalse(parameters.get('missing_param', False))

        parameters.cast_values = False
        self.assertEqual(parameters['myint']['value'], '1')
        self.assertEqual(parameters['mybool']['value'], 'true')

    def test_setting_valid_values(self):
        """Test that new parameters can be added"""
        parameters = Parameters(source=None, cast_values=False)
        self.assertEqual(len(parameters), 0)
        parameters['mystr'] = {
            "value": "hello",
            "section": "",
            "readableName": "",
            "helpTip": "",
            "recommended_values": "",
            "type": "str"
        }
        self.assertEqual(len(parameters), 1)

    def test_setting_invalid_values(self):
        """Test that directly setting invalid values raises an exception."""
        missing_help_tip = {
            "value": "true",
            "section": "",
            "readableName": "",
            "recommended_values": "",
            "type": "bool"
        }
        unsupported_type = {
            "value": "true",
            "section": "bci_config",
            "readableName": "Fake EEG Data On/Off",
            "helpTip": "",
            "recommended_values": "",
            "type": "custom_type"
        }

        parameters = Parameters(source=None, cast_values=False)
        self.assertEqual(len(parameters), 0)

        with self.assertRaises(Exception):
            parameters['test1'] = missing_help_tip

        with self.assertRaises(Exception):
            parameters['test2'] = unsupported_type

    def test_updating_uncast_values(self):
        """Test that new parameters can be added"""
        parameters = Parameters(source=None, cast_values=False)
        parameters['mystr'] = {
            "value": "hello",
            "section": "",
            "readableName": "",
            "helpTip": "",
            "recommended_values": "",
            "type": "str"
        }
        parameters['mystr']['value'] = 'hello world'
        self.assertEqual(parameters['mystr']['value'], 'hello world')

    def test_setting_existing_cast_values(self):
        """Test setting data when Parameters are cast"""
        sample_config = {
            "acq_port": {
                "value": "8000",
                "section": "acquisition",
                "readableName": "Acquisition Port",
                "helpTip": "",
                "recommended_values": "",
                "type": "int"
            },
            "acq_device": {
                "value": "LSL",
                "section": "acquisition",
                "readableName": "Acquisition Device",
                "helpTip": "",
                "recommended_values": ["LSL", "DSI"],
                "type": "str"
            },
            "is_txt_stim": {
                "value": "false",
                "section": "",
                "readableName": "",
                "helpTip": "",
                "recommended_values": "",
                "type": "bool"
            }
        }
        parameters = Parameters(source=None, cast_values=True)
        parameters.load(sample_config)
        self.assertEqual(parameters['is_txt_stim'], False)

        parameters['acq_port'] = 9000
        parameters['acq_device'] = 'DSI'
        parameters['is_txt_stim'] = True
        print(parameters)
        self.assertEqual(parameters['acq_port'], 9000)
        self.assertEqual(parameters['acq_device'], 'DSI')
        self.assertEqual(parameters['is_txt_stim'], True)

        parameters.cast_values = False
        self.assertEqual(parameters['acq_port']['value'], '9000')
        self.assertEqual(parameters['acq_device']['value'], 'DSI')
        self.assertEqual(parameters['is_txt_stim']['value'], 'true')

    def test_update(self):
        """Test update method"""
        sample_config = {
            "acq_port": {
                "value": "8000",
                "section": "acquisition",
                "readableName": "Acquisition Port",
                "helpTip": "",
                "recommended_values": "",
                "type": "int"
            },
            "acq_device": {
                "value": "LSL",
                "section": "acquisition",
                "readableName": "Acquisition Device",
                "helpTip": "",
                "recommended_values": ["LSL", "DSI"],
                "type": "str"
            }
        }
        parameters = Parameters(source=None, cast_values=True)
        parameters.load(sample_config)
        parameters.update(acq_port=9000, acq_device='DSI')

        self.assertEqual(parameters['acq_port'], 9000)
        self.assertEqual(parameters['acq_device'], 'DSI')

    def test_setting_missing_cast_values(self):
        """Test setting cast data for values that have not already been loaded."""
        parameters = Parameters(source=None, cast_values=True)
        with self.assertRaises(Exception):
            parameters['acq_port'] = 9000

    def test_values(self):
        """Test values method"""
        sample_config = {
            "acq_port": {
                "value": "8000",
                "section": "acquisition",
                "readableName": "Acquisition Port",
                "helpTip": "",
                "recommended_values": "",
                "type": "int"
            },
            "acq_device": {
                "value": "LSL",
                "section": "acquisition",
                "readableName": "Acquisition Device",
                "helpTip": "",
                "recommended_values": ["LSL", "DSI"],
                "type": "str"
            }
        }
        parameters = Parameters(source=None, cast_values=True)
        parameters.load(sample_config)
        self.assertEqual(list(parameters.keys()), ['acq_port', 'acq_device'])
        self.assertEqual(list(parameters.values()), [8000, 'LSL'])

    def test_save(self):
        """Test saving data to a json file"""

        parameters = Parameters(source=self.parameters_location)
        name = 'parameters_copy.json'
        parameters.save(directory=self.temp_dir, name=name)

        source = str(Path(self.temp_dir, name))
        params2 = Parameters(source=source, cast_values=False)

        self.assertEqual(parameters, params2)


if __name__ == '__main__':
    unittest.main()
