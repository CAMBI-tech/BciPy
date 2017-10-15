import os, errno
import datetime
from shutil import copy2

def init_save_data_structure(data_save_path, user_information, parameters_used):
	''' 
		data_save_path: string of path to save our data in
		user_information: string of user name / realted information
		parameters_used: a path to parameters file for the experiment

	'''

	# make an experiment folder : note datetime is in utc ('2013-11-18T08:18:31.809000')
	save_folder_name = data_save_path + user_information + datetime.datetime.now().isoformat()
	helper_folder_name = save_folder_name + '/helpers/'

	# try making the given path
	try:
		# make a directory to 
	    os.makedirs(save_folder_name)
	    os.makedirs(helper_folder_name)
	except OSError as e:
		# If the error is anything other than the file already existing, raise an error
	    if e.errno != errno.EEXIST:
	        raise


	# put in static things
	copy2(parameters_used, save_folder_name)

	# return path for completion or other data type saving needs
	return save_folder_name



def complete_save_data_structure(path):



	return path
