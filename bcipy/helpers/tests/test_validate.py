import os
import unittest

from bcipy.helpers.validate import validate_experiment
from bcipy.helpers.save import save_experiment_data
from bcipy.helpers.exceptions import (
    UnregisteredExperimentException,
    UnregisteredFieldException,
)
from bcipy.helpers.system_utils import DEFAULT_EXPERIMENT_ID


class TestValidateExperiment(unittest.TestCase):

    def test_validate_experiment_returns_true_on_default(self):
        experiment_name = DEFAULT_EXPERIMENT_ID

        # rely on the default experiment and field path kwargs, pass the experiment_name for validation
        response = validate_experiment(experiment_name)

        self.assertTrue(response)

    def test_validate_experiment_throws_unregistered_expection_on_unregistered_experiment(self):
        experiment_name = 'doesnotexist'
        with self.assertRaises(UnregisteredExperimentException):
            validate_experiment(experiment_name)

    def test_validate_experiment_throws_unregistered_exception_on_unregistered_fields(self):
        # create a fake experiment to load
        experiment_name = 'test'
        experiment = {
            experiment_name: {
                'fields': [{'does_not_exist': {'required': 'false'}}], 'summary': ''}
        }

        # save it to a custom path (away from default)
        path = save_experiment_data(experiment, '.', 'test_experiment.json')

        # assert it raises the expected exception
        with self.assertRaises(UnregisteredFieldException):
            validate_experiment(experiment_name, experiment_path=path)

        os.remove(path)

    def test_validate_experiment_throws_file_not_found_with_incorrect_experiment_path(self):
        # define an invalid path
        path = 'does/not/exist'

        # assert it raises the expected exception for an invalid experiment_path kwarg
        with self.assertRaises(FileNotFoundError):
            validate_experiment(DEFAULT_EXPERIMENT_ID, experiment_path=path)

    def test_validate_experiment_throws_file_not_found_with_incorrect_field_path(self):
        # define an invalid path
        path = 'does/not/exist'

        # assert it raises the expected exception for an invalid field_path kwarg
        with self.assertRaises(FileNotFoundError):
            validate_experiment(DEFAULT_EXPERIMENT_ID, field_path=path)
