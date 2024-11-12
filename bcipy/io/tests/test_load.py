import json
import os
import shutil
import tempfile
import unittest
from collections import abc
from unittest.mock import Mock, mock_open, patch

from mockito import any, expect, unstub, when

from bcipy.config import (DEFAULT_ENCODING, DEFAULT_EXPERIMENT_PATH,
                          DEFAULT_FIELD_PATH, DEFAULT_PARAMETERS_PATH,
                          EXPERIMENT_FILENAME, FIELD_FILENAME)
from bcipy.exceptions import BciPyCoreException, InvalidExperimentException
from bcipy.io.load import (choose_signal_model, choose_signal_models,
                           copy_parameters, extract_mode,
                           load_experiment_fields, load_experiments,
                           load_fields, load_json_parameters,
                           load_signal_model, load_users)
from bcipy.core.parameters import Parameters

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

    def tearDown(self):
        try:
            shutil.rmtree(self.data_save_loc)
        except FileNotFoundError:
            pass

    def test_user_load_with_no_directory_written(self):
        """Use defined data save location without writing anything"""
        response = load_users(self.data_save_loc)

        self.assertEqual(response, [])

    def test_user_load_with_valid_directory(self):
        user = 'user_001'
        file_path = f'{self.directory_name}/{user}/experiment'
        os.makedirs(file_path)

        response = load_users(self.data_save_loc)

        # There is only one user returned
        length_of_users = len(response)
        self.assertTrue(length_of_users == 1)

        # assert user returned is user defined above
        self.assertEqual(response[0], user)

    def test_user_load_with_invalid_directory(self):
        # create an invalid save structure and assert expected behavior.
        file_path = f'{self.directory_name}/'
        os.makedirs(file_path)

        response = load_users(self.data_save_loc)
        length_of_users = len(response)
        self.assertTrue(length_of_users == 0)


class TestExtractMode(unittest.TestCase):

    def test_extract_mode_calibration(self):
        data_save_path = 'data/user/default/user_RSVP_Calibration_Mon_01_Mar_2021_11hr19min49sec_-0800'
        expected_mode = 'calibration'
        response = extract_mode(data_save_path)
        self.assertEqual(expected_mode, response)

    def test_extract_mode_copy_phrase(self):
        data_save_path = 'data/user/default/user_RSVP_Copy_Phrase_Mon_01_Mar_2021_11hr19min49sec_-0800'
        expected_mode = 'copy_phrase'
        response = extract_mode(data_save_path)
        self.assertEqual(expected_mode, response)

    def test_extract_mode_without_mode_defined(self):
        invalid_data_save_dir = 'data/user/default/user_bad_dir'
        with self.assertRaises(BciPyCoreException):
            extract_mode(invalid_data_save_dir)


class TestModelLoad(unittest.TestCase):
    """Test loading one or more signal models"""

    @patch("bcipy.io.load.pickle.load")
    @patch("bcipy.io.load.open")
    def test_load_model(self, open_mock, pickle_mock):
        """Test loading a signal model"""

        load_signal_model("test-directory")
        open_mock.assert_called_with("test-directory", 'rb')
        pickle_mock.assert_called_once()

    @patch("bcipy.io.load.load_signal_model")
    @patch("bcipy.io.load.ask_filename")
    @patch("bcipy.io.load.preferences")
    def test_choose_model(self, preferences_mock, ask_file_mock,
                          load_signal_model_mock):
        """Test choosing a model"""

        preferences_mock.signal_model_directory = "."
        ask_file_mock.return_value = "model-path"
        model_mock = Mock()
        load_signal_model_mock.return_value = model_mock

        model = choose_signal_model('EEG')

        load_signal_model_mock.assert_called_with("model-path")
        ask_file_mock.assert_called_with(file_types="*.pkl",
                                         directory=".",
                                         prompt="Select the EEG signal model")
        self.assertEqual(model, model_mock)
        self.assertEqual("model-path",
                         preferences_mock.signal_model_directory,
                         msg="Should have updated the preferences")

    @patch("bcipy.io.load.load_signal_model")
    @patch("bcipy.io.load.ask_filename")
    @patch("bcipy.io.load.preferences")
    def test_choose_model_with_cancel(self, preferences_mock, ask_file_mock,
                                      load_signal_model_mock):
        """Test choosing a model"""

        preferences_mock.signal_model_directory = "."
        ask_file_mock.return_value = None
        model_mock = Mock()
        load_signal_model_mock.return_value = model_mock

        model = choose_signal_model('EEG')

        load_signal_model_mock.assert_not_called()
        ask_file_mock.assert_called_with(file_types="*.pkl",
                                         directory=".",
                                         prompt="Select the EEG signal model")
        self.assertEqual(None, model)
        self.assertEqual(".",
                         preferences_mock.signal_model_directory,
                         msg="Should not have updated the preferences")

    @patch("bcipy.io.load.choose_signal_model")
    def test_choose_signal_models(self, choose_signal_model_mock):
        """Test choosing signal models"""
        eeg_mock = Mock()
        eyetracker_mock = Mock()
        choose_signal_model_mock.side_effect = [eeg_mock, eyetracker_mock]

        models = choose_signal_models(['EEG', 'Eyetracker'])
        self.assertListEqual([eeg_mock, eyetracker_mock], models)

    @patch("bcipy.io.load.choose_signal_model")
    def test_choose_signal_models_missing_model(self,
                                                choose_signal_model_mock):
        """Test choosing signal models"""

        eyetracker_mock = Mock()
        choose_signal_model_mock.side_effect = [None, eyetracker_mock]

        models = choose_signal_models(['EEG', 'Eyetracker'])
        self.assertListEqual([eyetracker_mock], models)


if __name__ == '__main__':
    unittest.main()
