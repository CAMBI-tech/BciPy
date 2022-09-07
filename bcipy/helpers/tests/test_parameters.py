"""Tests for Parameter related functionality"""
import shutil
import tempfile
import unittest
from collections import abc
from pathlib import Path

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.helpers.parameters import Parameters


class TestParameters(unittest.TestCase):
    """Tests for loading and saving Parameters."""

    def setUp(self):
        """Override; set up the needed path for load functions."""

        self.parameters_location = DEFAULT_PARAMETERS_PATH
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override"""
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
        """Test load parameters initializes data."""
        parameters = Parameters(source=self.parameters_location)
        self.assertTrue(len(parameters) > 0)

    def test_initialization_without_source(self):
        """Test parameter initialization without a JSON source file."""

        parameters = Parameters(source=None)
        self.assertTrue(isinstance(parameters, abc.MutableMapping))
        self.assertEqual(len(parameters), 0)

    def test_load_data(self):
        """Test load data."""
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
        """Test cast_values."""

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
        self.assertEqual(parameters.get('myint', 0), 1,
                         'Should support `get` method with a default value')
        self.assertEqual(
            parameters.get('myint'), 1,
            'Should support `get` method without a default value')

        self.assertEqual(parameters['mybool'], True)
        self.assertEqual(parameters.get('mybool', False), True)
        self.assertEqual(parameters['mypath'],
                         'bcipy/parameters/parameters.json')
        self.assertEqual(parameters['mystr'], 'hello')
        self.assertEqual(parameters.get('mystr'), 'hello')
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

        parameters.cast_values = False
        for value in parameters.values():
            self.assertEqual(type(value), dict)

    def test_save_as(self):
        """Test saving data to a new json file"""

        parameters = Parameters(source=self.parameters_location)
        name = 'parameters_copy.json'
        parameters.save(directory=self.temp_dir, name=name)

        source = str(Path(self.temp_dir, name))
        params2 = Parameters(source=source, cast_values=False)

        self.assertEqual(parameters, params2)

    def test_save(self):
        """Test saving to overwrite source file."""

        # Save a copy to the temp directory
        parameters = Parameters(source=self.parameters_location,
                                cast_values=True)
        name = 'parameters_copy.json'
        parameters.save(directory=self.temp_dir, name=name)

        source = str(Path(self.temp_dir, name))
        params2 = Parameters(source=source, cast_values=True)
        self.assertEqual(parameters, params2)

        # Change a parameter and overwrite
        self.assertTrue(parameters['parameter_location'] != source)
        params2['parameter_location'] = source
        params2.save()

        # Load from the save file and confirm that saved values persisted
        params3 = Parameters(source=source, cast_values=True)
        self.assertEqual(params3['parameter_location'], source)
        self.assertEqual(params3, params2)
        self.assertNotEqual(params3, parameters)

    def test_save_new(self):
        """Test saving a new params file."""
        parameters = Parameters(source=None, cast_values=False)
        parameters['mystr'] = {
            "value": "hello",
            "section": "",
            "readableName": "",
            "helpTip": "",
            "recommended_values": "",
            "type": "str"
        }
        with self.assertRaises(Exception):
            # Missing directory and file name
            parameters.save()

        with self.assertRaises(Exception):
            parameters.save(directory=self.temp_dir)
        with self.assertRaises(Exception):
            parameters.save(name='my_params.json')

        parameters.save(directory=self.temp_dir, name='my_params.json')

    def test_items(self):
        """Test items"""
        parameters = Parameters(source=None, cast_values=False)
        parameters['mystr'] = {
            "value": "hello",
            "section": "",
            "readableName": "",
            "helpTip": "",
            "recommended_values": "",
            "type": "str"
        }
        self.assertEqual(len(parameters.items()), 1)
        for key, val in parameters.items():
            self.assertEqual(key, 'mystr')
            self.assertEqual(type(val), dict)

        parameters.cast_values = True
        self.assertEqual(len(parameters.items()), 1)
        for key, val in parameters.items():
            self.assertEqual(key, 'mystr')
            self.assertEqual(val, 'hello')

    def test_copy(self):
        """Test copy"""
        parameters = Parameters(source=self.parameters_location,
                                cast_values=True)
        params_copy = parameters.copy()

        self.assertEqual(params_copy.source, None)
        self.assertEqual(params_copy.cast_values, parameters.cast_values)
        self.assertEqual(params_copy.values(), parameters.values())

    def test_check_entry(self):
        parameters = Parameters(source=None, cast_values=False)
        parameters.check_valid_entry(
            "fake_data", {
                "value": "true",
                "section": "bci_config",
                "readableName": "Fake Data Sessions",
                "helpTip": "If true, fake data server used",
                "recommended_values": "",
                "type": "bool"
            })
        with self.assertRaises(Exception):
            parameters.check_valid_entry("fake_data", True)

    def test_check_entry_bool_type(self):
        """Test that invalid bool types are rejected"""
        parameters = Parameters(source=None, cast_values=False)
        with self.assertRaises(Exception):
            parameters.check_valid_entry(
                "fake_data", {
                    "value": True,
                    "section": "bci_config",
                    "readableName": "Fake Data Sessions",
                    "helpTip": "If true, fake data server used",
                    "recommended_values": "",
                    "type": "bool"
                })

    def test_alternate_constructor(self):
        """Test alternate constructor from cast values"""
        parameters = Parameters.from_cast_values(myint=1,
                                                 mybool=True,
                                                 mystr="Testing")
        self.assertTrue(parameters.cast_values)
        self.assertEqual(parameters['myint'], 1)
        self.assertEqual(parameters['mybool'], True)
        self.assertEqual(parameters['mystr'], 'Testing')

        parameters.cast_values = False
        self.assertEqual(parameters['myint']['value'], '1')
        self.assertEqual(parameters['mybool']['value'], 'true')
        self.assertEqual(parameters['mystr']['value'], 'Testing')

    def test_add_missing(self):
        """Test add_missing_items"""
        entry1 = {
            "value": "8000",
            "section": "acquisition",
            "readableName": "Acquisition Port",
            "helpTip": "",
            "recommended_values": "",
            "type": "int"
        }
        entry2 = {
            "value": "LSL",
            "section": "acquisition",
            "readableName": "Acquisition Device",
            "helpTip": "",
            "recommended_values": ["LSL", "DSI"],
            "type": "str"
        }

        parameters = Parameters(source=None)
        parameters.load({"acq_port": entry1})

        new_params = Parameters(source=None)
        new_params.load({"acq_port": entry1, "acq_device": entry2})

        self.assertFalse('acq_device' in parameters.keys())
        self.assertTrue(parameters.add_missing_items(new_params))
        self.assertTrue('acq_device' in parameters.keys())

        self.assertFalse(parameters.add_missing_items(new_params),
                         "Only new parameters should be added.")

    def test_changed_parameters(self):
        """Test diff calculations"""
        entry1 = {
            "value": "8000",
            "section": "acquisition",
            "readableName": "Acquisition Port",
            "helpTip": "",
            "recommended_values": "",
            "type": "int"
        }
        entry2 = {
            "value": "75E+6",
            "section": "artifact_rejection",
            "readableName": "High Voltage Threshold Value",
            "helpTip": "Specifies the high voltage threshold (in microvolts)",
            "recommended_values": "",
            "type": "float"
        }
        entry2_same = {
            "value": "75000000.0",
            "section": "artifact_rejection",
            "readableName": "High Voltage Threshold Value",
            "helpTip": "Specifies the high voltage threshold (in microvolts)",
            "recommended_values": "",
            "type": "float"
        }
        entry3 = {
            "value": "DSI-24",
            "section": "acquisition",
            "readableName": "Acquisition Device",
            "helpTip": "",
            "recommended_values": ["DSI-24", "DSI-VR300"],
            "type": "str"
        }
        entry3_modified = {
            "value": "DSI-VR300",
            "section": "acquisition",
            "readableName": "Acquisition Device",
            "helpTip": "",
            "recommended_values": ["DSI-24", "DSI-VR300"],
            "type": "str"
        }
        parameters = Parameters(source=None)
        parameters.load({
            "entry_1": entry1,
            "entry_2": entry2,
            "entry_3": entry3
        })

        new_params = Parameters(source=None)
        new_params.load({
            "entry_1": entry1,
            "entry_2": entry2_same,
            "entry_3": entry3_modified
        })
        changes = new_params.diff(parameters)

        self.assertEqual(len(changes.keys()), 1)
        self.assertTrue("entry_3" in changes.keys())
        self.assertEqual(changes["entry_3"].original_value, entry3['value'])
        self.assertEqual(changes["entry_3"].parameter['value'],
                         entry3_modified['value'])


if __name__ == '__main__':
    unittest.main()
