import errno
import os
import shutil
import time
import unittest

from bcipy.helpers.save import init_save_data_structure


class TestSave(unittest.TestCase):
    """This is Test Case for Saving BCI data."""

    def setUp(self):
        # set up the needed paths and initial data save structure

        self.data_save_path = 'data/'
        self.user_information = 'test_user_002'
        self.parameters_used = 'bcipy/parameters/parameters.json'

        self.save_folder_name = init_save_data_structure(
            self.data_save_path,
            self.user_information,
            self.parameters_used)

    def tearDown(self):
        # clean up by removing the data folder we used for testing
        shutil.rmtree(self.data_save_path)

    def test_init_save_data_structure_creates_correct_save_folder(self):

        # assert the save folder was created
        self.assertTrue(os.path.isdir(self.save_folder_name))

    def test_parameter_file_copies(self):

        # construct the path of the parameters
        param_path = self.save_folder_name + "/parameters.json"

        # assert that the params file was created in the correct location
        self.assertTrue(os.path.isfile(param_path))

    def test_throws_useful_error_if_given_incorrect_params_path(self):

        # try passing a parameters file that does not exist
        with self.assertRaises(Exception):
            init_save_data_structure(
                self.data_save_path,
                'new_user',
                'does_not_exist.json')

    def test_init_save_data_structure_makes_helpers_folder(self):

        # contruct the path of the helper folder
        helper_folder_name = self.save_folder_name + '/helpers/'

        # attempt to make that folder
        try:
            os.makedirs(helper_folder_name)

        except OSError as error:
            # assert the error returned, is that the dir exists.
            self.assertEqual(error.errno, errno.EEXIST)

if __name__ == '__main__':
    unittest.main()