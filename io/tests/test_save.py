import datetime, errno, os, pytest, shutil, sys, unittest

# add previous dirs to python path for importing of modules
sys.path.append('..')

from save import init_save_data_structure


class TestSave(unittest.TestCase):

	def setUp(self):
		# set up the needed paths and initial data save structure

		self.data_save_path = 'test_files/data/'
		self.user_information = 'test_user_001'
		self.parameters_used = 'test_files/parameters.json'

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


	def test_init_save_data_structure_moves_parameters_file(self):

		# construct the path of the parameters
		param_path = self.save_folder_name + "/parameters.json"

		# assert that the params file was created in the correct location
		self.assertTrue(os.path.isfile(param_path))

	def test_init_save_data_structure_throws_useful_error_if_given_incorrect_params_path(self):

		# try passing a parameters file that does not exist
		try:
			save_folder_name = init_save_data_structure(
				self.data_save_path,
				self.user_information,
				'doesnotexit.json')

		# catch the exception and make sure it's as expected
		except Exception as error:
			self.assertEqual(error.strerror, 'No such file or directory')


	def test_init_save_data_structure_makes_helpers_folder(self):

		# contruct the path of the helper folder
		helper_folder_name = self.save_folder_name + '/helpers/'

		# attempt to make that folder
		try:
		    os.makedirs(helper_folder_name)

		except OSError as error:
			# assert the error returned, is that the dir exists.
		    self.assertEqual(error.errno, errno.EEXIST)

