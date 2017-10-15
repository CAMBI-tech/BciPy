import os
import pytest
import sys
import datetime
sys.path.append('..')
sys.path.append('.')
from save import init_save_data_structure

def test_init_save_data_structure():

	# mock the necessary inputs for initializing the save data function
	data_save_path = 'data/'
	user_information = 'user'
	parameters_used = '../gui/utility/parameters.json'


	save_folder_name = init_save_data_structure(
		data_save_path,
		user_information,
		parameters_used)


	assert save_folder_name == save_folder_name

# test_init_save_data_structure()