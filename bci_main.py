# import acquisition
from gui.utility.gui_fx import run_python_file
from helpers.save import init_save_data_structure
from helpers.display import init_display_window
# from helpers.init import init_daq, init_display
# from utils import *
from bci_tasks.start_task import start_task
import pdb


def bci_main(parameters, user, exp_type, mode):

    # Initalize Save Folder
    save_folder = init_save_data_structure(
        'data/', user, 'parameters/parameters.json')
    # Execute Task

    task_type = {
        'mode': mode,
        'exp_type': exp_type
    }

    trial_data = execute_task(
        task_type, parameters, save_folder)
    # Finalize Save

    # Open Gui
    # run_python_file('gui/BCInterface.py')

    print "Successful Trial %s" % (trial_data)


def execute_task(task_type, parameters, save_folder):

    # # Initialize DAQ
    # daq = init_daq()

    daq = "TEST DAQ"

    # # Initialize Window
    display = init_display_window(parameters)

    # Start Task
    try:
        trial_data = start_task(
            daq, display, task_type, parameters, save_folder)
    except Exception as e:
        pdb.set_trace()

    display.close()
    # return trial_data
    return trial_data
