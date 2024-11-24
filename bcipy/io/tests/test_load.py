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
                           load_bcipy_data,
                           load_fields, load_json_parameters,
                           load_signal_model, load_users)
from bcipy.io.tests.test_convert import create_bcipy_session_artifacts
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


class TestLoadBciPyData(unittest.TestCase):
    """data_directory/
        user_ids/
            dates/
                experiment_ids/
                    datetimes/
                        protocol.json
                        logs/
                        tasks/
                            *task_data*"""

    def setUp(self):
        # make a temporary directory mimicking the structure of the data directory
        self.data_dir = tempfile.mkdtemp()
        self.user_ids = ['user1', 'user2']
        self.dates = ['2024-10-31', '2024-10-32']
        self.experiment_ids = ['experiment1', 'experiment2']
        self.datetimes = ['2024-10-31_15-01-05', '2024-10-32_15-01-07']
        self.tasks = ['task1', 'task2']

        # create the directory structure
        for user_id in self.user_ids:
            user_dir = os.path.join(self.data_dir, user_id)
            os.makedirs(user_dir)
            for date in self.dates:
                date_dir = os.path.join(user_dir, date)
                os.makedirs(date_dir)
                for experiment_id in self.experiment_ids:
                    experiment_dir = os.path.join(date_dir, experiment_id)
                    os.makedirs(experiment_dir)
                    for datetime in self.datetimes:
                        datetime_dir = os.path.join(experiment_dir, datetime)
                        os.makedirs(datetime_dir)
                        for task in self.tasks:
                            task_dir = os.path.join(datetime_dir, task)
                            os.makedirs(task_dir)
                            # create artificial files
                            create_bcipy_session_artifacts(task_dir)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_load_bcipy_data(self):
        total_expected_files = len(self.user_ids) * len(self.dates) * len(self.experiment_ids) * \
            len(self.datetimes) * len(self.tasks)
        response = load_bcipy_data(self.data_dir)
        self.assertEqual(len(response), total_expected_files)

    def test_load_bcipy_data_with_invalid_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_bcipy_data('')

    def test_load_bcipy_data_with_user_id_filter(self):
        # pick one user id to filter
        user_id = self.user_ids[0]
        total_expected_files = len(self.dates) * len(self.experiment_ids) * \
            len(self.datetimes) * len(self.tasks)
        response = load_bcipy_data(self.data_dir, user_id=user_id)
        self.assertEqual(len(response), total_expected_files)

    def test_load_bcipy_data_with_task_filter(self):
        excluded_task = [self.tasks[0]]
        total_expected_files = len(self.user_ids) * len(self.dates) * len(self.experiment_ids) * \
            len(self.datetimes) * (len(self.tasks) - 1)
        response = load_bcipy_data(self.data_dir, excluded_tasks=excluded_task)
        self.assertEqual(len(response), total_expected_files)

    def test_load_bcipy_data_with_date_filter(self):
        desired_date = self.dates[0]

        total_expected_files = len(self.user_ids) * (len(self.dates) - 1) * len(self.experiment_ids) * \
            len(self.datetimes) * len(self.tasks)

        response = load_bcipy_data(self.data_dir, date=desired_date)
        self.assertEqual(len(response), total_expected_files)
        # check that the date is in the response
        dates = [file.date for file in response]
        self.assertIn(desired_date, dates)

    def test_load_bcipy_data_with_experiment_id_filter(self):
        desired_experiment_id = self.experiment_ids[0]

        total_expected_files = len(self.user_ids) * len(self.dates) * (len(self.experiment_ids) - 1) * \
            len(self.datetimes) * len(self.tasks)
        response = load_bcipy_data(self.data_dir, experiment_id=desired_experiment_id)

        self.assertEqual(len(response), total_expected_files)
        experiments = [file.experiment_id for file in response]
        # check that the experiment id is in the response
        self.assertIn(desired_experiment_id, experiments)
        self.assertNotIn(self.experiment_ids[1], experiments)

    def test_load_bcipy_data_with_datetime_filter(self):
        desired_datetime = self.datetimes[0]

        total_expected_files = len(self.user_ids) * len(self.dates) * len(self.experiment_ids) * \
            (len(self.datetimes) - 1) * len(self.tasks)
        response = load_bcipy_data(self.data_dir, date_time=desired_datetime)

        self.assertEqual(len(response), total_expected_files)
        # check that the datetime is in the response
        datetimes = [file.date_time for file in response]
        self.assertIn(desired_datetime, datetimes)
        self.assertNotIn(self.datetimes[1], datetimes)




if __name__ == '__main__':
    unittest.main()
