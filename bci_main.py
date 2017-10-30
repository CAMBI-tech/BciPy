import gui.utility.gui_fx
from helpers.save import init_save_data_structure
from helpers.display import init_display_window
from bci_tasks.start_task import start_task


def bci_main(parameters, user, exp_type, mode):
    """
    BCI Main.

    Using parameters (dict), user information, exp_type
        ( ex. calibration v. free spell), and mode ( ex .RSVP v. SSVEP)
        the BCI app will initialize a save folder, construct needed information
        and execute the task. This is the main connection between any UI and
        running the app.
    """

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

    return "Successful Trial"


def execute_task(task_type, parameters, save_folder):
    """
    Excecute Task.

    Executes the desired task by setting up the display window and
        data acquistion, then passing on to the start_task funtion
        which will initialize experiment.
    """

    # Initialize DAQ [TO-DO: MAKE INIT_DAQ FUNCTION]
    # daq = init_daq()

    daq = "TEST DAQ"

    # # Initialize Window
    display = init_display_window(parameters)

    # Start Task
    try:
        trial_data = start_task(
            daq, display, task_type, parameters, save_folder)
    except Exception as e:
        print e

    display.close()

    # return trial_data
    return trial_data
