import os
import shutil
import unittest
import tempfile

from bcipy.config import DEFAULT_PARAMETERS_PATH, DEFAULT_PARAMETER_FILENAME, DEFAULT_EXPERIMENT_ID
from bcipy.helpers import save
from bcipy.helpers.save import init_save_data_structure
from mockito import any, unstub, when


class TestSave(unittest.TestCase):
    """This is Test Case for Saving BCI data."""

    def setUp(self):
        # set up the needed paths and initial data save structure

        self.data_save_path = tempfile.mkdtemp()
        self.user_information = 'test_user_002'
        self.parameters_used = DEFAULT_PARAMETERS_PATH
        self.task = 'RSVP Calibration'
        self.experiment = DEFAULT_EXPERIMENT_ID

        self.save_folder_name = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            self.parameters_used,
            self.task)

        self.dt = '10'

        # mock save modules use of strftime to return an empty string
        when(save).strftime(any(), any()).thenReturn(self.dt)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.data_save_path)
        unstub()

    def test_init_save_data_structure_creates_correct_save_folder(self):

        # assert the save folder was created
        self.assertTrue(os.path.isdir(self.save_folder_name))

    def test_parameter_file_copies(self):

        # construct the path of the parameters
        param_path = self.save_folder_name + f'/{DEFAULT_PARAMETER_FILENAME}'

        # assert that the params file was created in the correct location
        self.assertTrue(os.path.isfile(param_path))

    def test_save_structure_adds_default_experiment(self):
        self.assertIn(DEFAULT_EXPERIMENT_ID, self.save_folder_name)

    def test_save_structure_adds_experiment_id_when_provided_as_argument(self):
        experiment_id = 'test'
        response = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            self.parameters_used,
            self.task,
            experiment_id=experiment_id)
        task = self.task.replace(' ', '_')
        expected = (
            f'{self.data_save_path}{experiment_id}'
            f'/{self.user_information}/{self.user_information}_{task}_{self.dt}'
        )

        self.assertEqual(response, expected)

    def test_throws_error_if_given_incorrect_params_path(self):

        # try passing a parameters file that does not exist
        with self.assertRaises(Exception):
            init_save_data_structure(
                self.data_save_path,
                self.user_information,
                'does_not_exist.json',
                self.task,
                self.experiment)


if __name__ == '__main__':
    unittest.main()
