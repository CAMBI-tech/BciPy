# import acquisition
from gui.utility.gui_fx import run_python_file
from helpers.save import init_save_data_structure
# from helpers.init import init_daq, init_display
# from utils import *
from bci_tasks import start_task


def bci_main(parameters, user, task_type):

    # Initalize Save Folder
    save_folder = init_save_data_structure(
        'data/', user, 'parameters/parameters.json')
    # Execute Task

    trial_data = execute_task(
        task_type, parameters, save_folder)
    # Finalize Save

    # Open Gui
    run_python_file('gui/BCInterface.py')


def execute_task(task_type, parameters, save_folder):

    # # Initialize DAQ
    # daq = init_daq()

    # # Initialize Window
    # display = init_display()

    # Start Task
    # trial_data = start_task(daq, display, task_type, parameters, save_folder)

    # return trial_data
    print "execute_task"
