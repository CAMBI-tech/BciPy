import unittest
from unittest.mock import patch, mock_open

from collections import abc
import tempfile
import shutil
import pickle
import json

from mockito import any, expect, unstub, when

from bcipy.helpers.load import (
    load_json_parameters,
    load_signal_model,
    load_experiments,
    load_experiment_fields,
    load_fields,
    copy_parameters)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.exceptions import InvalidExperimentException
from bcipy.helpers.system_utils import DEFAULT_EXPERIMENT_PATH, DEFAULT_FIELD_PATH, FIELD_FILENAME, EXPERIMENT_FILENAME


MOCK_EXPERIMENT = {
    "test": {
        "fields": [
            {
                "age": {
                    "required": "false"
                }
            }
        ],
        "summary": "test summary"
    }
}


class TestParameterLoad(unittest.TestCase):
    """This is Test Case for Loading BCI data."""

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters = 'bcipy/parameters/parameters.json'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_json_parameters_returns_dict(self):
        """Test load parameters returns a Python dict."""

        # call the load parameters function
        parameters = load_json_parameters(self.parameters)

        # assert that load function turned json parameters into a dict-like obj
        self.assertEqual(type(parameters), Parameters)
        self.assertTrue(isinstance(parameters, abc.MutableMapping))

    def test_load_json_parameters_throws_error_on_wrong_path(self):
        """Test load parameters returns error on entering wrong path."""

        # call the load parameters function with incorrect path
        with self.assertRaises(Exception):
            load_json_parameters('/garbage/dir/wont/work')

    def test_load_classifier(self):
        """Test load classifier can load pickled file when given path."""

        # create a pickle file to save a pickled json
        pickle_file = self.temp_dir + "save.p"
        pickle.dump(self.parameters, open(pickle_file, "wb"))

        # Load classifier
        unpickled_parameters = load_signal_model(pickle_file)

        # assert the same data was returned
        self.assertEqual(unpickled_parameters, (self.parameters, pickle_file))

    def test_copy_default_parameters(self):
        """Test that default parameters can be copied."""
        path = copy_parameters(destination=self.temp_dir)

        self.assertTrue(path != self.parameters)

        copy = load_json_parameters(path)
        self.assertTrue(type(copy), 'dict')

        parameters = load_json_parameters(self.parameters)
        self.assertEqual(copy, parameters)


class TestExperimentLoad(unittest.TestCase):

    def setUp(self):
        self.experiments_path = f'{DEFAULT_EXPERIMENT_PATH}{EXPERIMENT_FILENAME}'
        when(json).load(any()).thenReturn()

    def tearDown(self):
        unstub()

    def test_load_experiments_calls_open_with_expected_default(self):
        with patch('builtins.open', mock_open(read_data='data')) as mock_file:
            load_experiments()
            mock_file.assert_called_with(self.experiments_path, 'r')

    def test_load_experiments_throws_file_not_found_exception_with_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_experiments(path='')

    def test_load_experiments_calls_json_module_as_expected(self):
        with patch('builtins.open', mock_open(read_data='data')) as mock_file:
            expect(json, times=1).loads(self.experiments_path)
            load_experiments()


class TestFieldLoad(unittest.TestCase):
    def setUp(self):
        self.fields_path = f'{DEFAULT_FIELD_PATH}{FIELD_FILENAME}'
        when(json).load(any()).thenReturn()

    def tearDown(self):
        unstub()

    def test_load_fields_calls_open_with_expected_default(self):
        with patch('builtins.open', mock_open(read_data='data')) as mock_file:
            load_fields()
            mock_file.assert_called_with(self.fields_path, 'r')

    def test_load_fields_throws_file_not_found_exception_with_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_fields(path='')

    def test_load_fields_calls_json_module_as_expected(self):
        with patch('builtins.open', mock_open(read_data='data')) as mock_file:
            expect(json, times=1).loads(self.fields_path)
            load_fields()


class TestExperimentFieldLoad(unittest.TestCase):

    def setUp(self):
        self.experiment = MOCK_EXPERIMENT['test']

    def test_load_experiment_fields_returns_a_list(self):
        fields = load_experiment_fields(self.experiment)
        self.assertIsInstance(fields, list)

    def test_load_experiment_fields_raises_type_error_on_non_dict_experiment(self):
        invalid_experiment_type = ''
        with self.assertRaises(TypeError):
            load_experiment_fields(invalid_experiment_type)

    def test_load_experiment_fields_raises_invalid_experiment_on_incorrectly_formatted_experiment(self):
        # create an experiment with the wrong field key
        invalid_experiment_field_name = {
            'summary': 'blah',
            'field': []
        }
        with self.assertRaises(InvalidExperimentException):
            load_experiment_fields(invalid_experiment_field_name)


if __name__ == '__main__':
    unittest.main()
