import acquisition
from gui import BCInterface
from helpers import init_save_data_structure, init_daq, init_display
from utils import *
from bci_tasks import start_task

def bci_main(parameters, user):
	
	# Initalize Save Folder
	save_folder = init_save_data_structure(
		parameters['data_save_path'], user, 'parameters/parameters.json')
	# Execute Task

	trial_data = execute_task(parameters['task_type'], parameters, save_folder)
	# Finalize Save

	# Open Gui 
	BCInterface()


def execute_task(task_type, parameters, save_folder):

	# Initialize DAQ
	daq = init_daq()

	# Initialize Window
	display = init_display() 

	# Start Task
	trial_data = start_task(daq, display, task_type, parameters, save_folder)
	return trial_data