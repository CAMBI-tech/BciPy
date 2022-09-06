import unittest
import os
from unittest.mock import patch, mock_open

from collections import abc
import tempfile
import shutil
import json

from mockito import any, expect, unstub, when

from bcipy.config import (
    DEFAULT_ENCODING,
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_PARAMETERS_PATH,
    DEFAULT_FIELD_PATH,
    FIELD_FILENAME,
    EXPERIMENT_FILENAME
)
from bcipy.helpers.load import (
    extract_mode,
    load_json_parameters,
    load_experiments,
    load_experiment_fields,
    load_fields,
    load_users,
    copy_parameters)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.exceptions import BciPyCoreException, InvalidExperimentException


MOCK_EXPERIMENT = {
    "test": {
        "fields": [
            {
                "age": {
                    "required": "false",
                    "anonymize": "true"
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

        self.parameters = DEFAULT_PARAMETERS_PATH
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
        self.experiments_path = f'{DEFAULT_EXPERIMENT_PATH}/{EXPERIMENT_FILENAME}'
        when(json).load(any()).thenReturn()

    def tearDown(self):
        unstub()

    def test_load_experiments_calls_open_with_expected_default(self):
        with patch('builtins.open', mock_open(read_data='data')) as mock_file:
            load_experiments()
            mock_file.assert_called_with(self.experiments_path, 'r', encoding=DEFAULT_ENCODING)

    def test_load_experiments_throws_file_not_found_exception_with_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_experiments(path='')

    def test_load_experiments_calls_json_module_as_expected(self):
        with patch('builtins.open', mock_open(read_data='data')) as _:
            expect(json, times=1).loads(self.experiments_path)
            load_experiments()


class TestFieldLoad(unittest.TestCase):
    def setUp(self):
        self.fields_path = f'{DEFAULT_FIELD_PATH}/{FIELD_FILENAME}'
        when(json).load(any()).thenReturn()

    def tearDown(self):
        unstub()

    def test_load_fields_calls_open_with_expected_default(self):
        with patch('builtins.open', mock_open(read_data='data')) as mock_file:
            load_fields()
            mock_file.assert_called_with(self.fields_path, 'r', encoding=DEFAULT_ENCODING)

    def test_load_fields_throws_file_not_found_exception_with_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_fields(path='')

    def test_load_fields_calls_json_module_as_expected(self):
        with patch('builtins.open', mock_open(read_data='data')) as _:
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


class TestUserLoad(unittest.TestCase):

    def setUp(self):
        # setup parameters to pass to load users method, it expects a key of data_save_loc only
        self.directory_name = 'test_data_load_user'
        self.data_save_loc = f'{self.directory_name}/'

    def test_user_load_with_no_directory_written(self):
        """Use defined data save location without writing anything"""
        response = load_users(self.data_save_loc)

        self.assertEqual(response, [])

    def test_user_load_with_valid_directory(self):
        user = 'user_001'
        file_path = f'{self.directory_name}/experiment/{user}'
        os.makedirs(file_path)

        response = load_users(self.data_save_loc)

        # There is only one user returned
        length_of_users = len(response)
        self.assertTrue(length_of_users == 1)

        # assert user returned is user defined above
        self.assertEqual(response[0], user)
        shutil.rmtree(self.data_save_loc)

    def test_user_load_with_invalid_directory(self):
        # create an invalid save structure and assert expected behavior.
        user = 'user_001'
        file_path = f'{self.directory_name}/experiment{user}'
        os.makedirs(file_path)

        response = load_users(self.data_save_loc)
        length_of_users = len(response)
        self.assertTrue(length_of_users == 0)
        shutil.rmtree(self.data_save_loc)


class TestExtractMode(unittest.TestCase):

    def test_extract_mode_calibration(self):
        data_save_path = 'data/default/user/user_RSVP_Calibration_Mon_01_Mar_2021_11hr19min49sec_-0800'
        expected_mode = 'calibration'
        response = extract_mode(data_save_path)
        self.assertEqual(expected_mode, response)

    def test_extract_mode_copy_phrase(self):
        data_save_path = 'data/default/user/user_RSVP_Copy_Phrase_Mon_01_Mar_2021_11hr19min49sec_-0800'
        expected_mode = 'copy_phrase'
        response = extract_mode(data_save_path)
        self.assertEqual(expected_mode, response)

    def test_extract_mode_without_mode_defined(self):
        invalid_data_save_dir = 'data/default/user/user_bad_dir'
        with self.assertRaises(BciPyCoreException):
            extract_mode(invalid_data_save_dir)


if __name__ == '__main__':
    unittest.main()
