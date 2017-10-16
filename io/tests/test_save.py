import os, errno, sys
import pytest
import datetime
import unittest
import shutil
sys.path.append('....')
sys.path.append('..')
sys.path.append('.')
from save import init_save_data_structure



class TestSave(unittest.TestCase):

	def setUp(self):

		self.data_save_path = 'test_files/data/'
		self.user_information = 'test_user_001'
		self.parameters_used = 'test_files/parameters.json'

		self.save_folder_name = init_save_data_structure(
			self.data_save_path,
			self.user_information,
			self.parameters_used)

	def tearDown(self):

		shutil.rmtree(self.data_save_path)

	def test_init_save_data_structure_creates_correct_save_folder(self):

		# # mock the necessary inputs for initializing the save data function

		# make sure that it returns the root folder for later usage. Other data material
			# should be saved below this.

		self.assertTrue(os.path.isdir(self.save_folder_name))


	def test_init_save_data_structure_moves_parameters_file(self):

		# make sure that it returns the root folder for later usage. Other data material
			# should be saved below this.

		param_path = self.save_folder_name + "/parameters.json"

		self.assertTrue(os.path.isfile(param_path))


	def test_init_save_data_structure_makes_helpers_folder(self):


		helper_folder_name = self.save_folder_name + '/helpers/'

		try:
		    os.makedirs(helper_folder_name)
		except OSError as e:
			# If the error is anything other than the file already existing, raise an error
		    assert e.errno == errno.EEXIST


