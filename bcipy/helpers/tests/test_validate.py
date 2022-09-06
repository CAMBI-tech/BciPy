import unittest

from bcipy.config import DEFAULT_EXPERIMENT_ID
from bcipy.helpers.validate import validate_experiment, validate_experiments
from bcipy.helpers.save import save_experiment_data
from bcipy.helpers.exceptions import (
    InvalidExperimentException,
    InvalidFieldException,
    UnregisteredExperimentException,
    UnregisteredFieldException,
)


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

    def test_save_experiment_data_throws_unregistered_exception_on_unregistered_fields(self):
        # create a fake experiment to load
        experiment_name = 'test'
        fields = {
            'registered_field': {
                'help_text': 'test',
                'type': 'int'
            }
        }
        experiment = {
            experiment_name: {
                'fields': [{'does_not_exist': {'required': 'false', 'anonymize': 'true'}}], 'summary': ''}
        }

        # assert it raises the expected exception
        with self.assertRaises(UnregisteredFieldException):
            save_experiment_data(experiment, fields, '.', 'test_experiment.json')

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


class TestValidateExperiments(unittest.TestCase):
    experiment_name = 'test'
    fields = {
        'registered_field': {
            'help_text': 'test',
            'type': 'int'
        }
    }
    experiments = {
        experiment_name: {
            'fields': [{'registered_field': {'required': 'false', 'anonymize': 'true'}}], 'summary': ''}
    }

    def test_validate_experiments_returns_true_on_valid_experiment(self):
        response = validate_experiments(self.experiments, self.fields)

        self.assertTrue(response)

    def test_validate_experiments_throws_invalid_experiment_exception_on_invalid_experiment_no_field(self):
        experiments = {
            'invalid': {
                'summary': ''}
        }
        with self.assertRaises(InvalidExperimentException):
            validate_experiments(experiments, self.fields)

    def test_validate_experiments_throws_invalid_experiment_exception_on_invalid_experiment_invalid_field(self):
        experiments = {
            'invalid': {
                'summary': '',
                'fields': 'should_be_list!'}
        }
        with self.assertRaises(InvalidExperimentException):
            validate_experiments(experiments, self.fields)

    def test_validate_experiments_throws_invalid_experiment_exception_on_invalid_experiment_invalid_summary(self):
        experiments = {
            'invalid': {
                'summary': [],
                'fields': []}
        }
        with self.assertRaises(InvalidExperimentException):
            validate_experiments(experiments, self.fields)

    def test_validate_experiments_throws_invalid_experiment_exception_on_invalid_experiment_no_summary(self):
        experiments = {
            'invalid': {
                'fields': []}
        }
        with self.assertRaises(InvalidExperimentException):
            validate_experiments(experiments, self.fields)

    def test_validate_experiments_throws_invalid_field_exception_on_invalid_field_no_required(self):
        experiments = {
            self.experiment_name: {
                'fields': [{'registered_field': {'anonymize': 'true'}}], 'summary': ''}
        }
        with self.assertRaises(InvalidFieldException):
            validate_experiments(experiments, self.fields)

    def test_validate_experiments_throws_invalid_field_exception_on_invalid_field_no_anonymize(self):
        experiments = {
            self.experiment_name: {
                'fields': [{'registered_field': {'required': 'true'}}], 'summary': ''}
        }
        with self.assertRaises(InvalidFieldException):
            validate_experiments(experiments, self.fields)

    def test_validate_experiments_throws_unregistered_exception_on_unregistered_fields(self):
        experiment = {
            self.experiment_name: {
                'fields': [{'does_not_exist': {'required': 'false', 'anonymize': 'true'}}], 'summary': ''}
        }

        # assert it raises the expected exception
        with self.assertRaises(UnregisteredFieldException):
            validate_experiments(experiment, self.fields)


if __name__ == '__main__':
    unittest.main()
