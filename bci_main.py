import logging
from bcipy.display.display_main import init_display_window
from bcipy.helpers.acquisition import init_eeg_acquisition
from bcipy.helpers.task import print_message
from bcipy.helpers.system_utils import get_system_info, configure_logger
from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import load_json_parameters, load_signal_model
from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH
from bcipy.helpers.save import init_save_data_structure, DEFAULT_EXPERIMENT_ID
from bcipy.tasks.start_task import start_task
from bcipy.tasks.task_registry import TaskType


def bci_main(parameter_location: str, user: str, task: TaskType, experiment: str = DEFAULT_EXPERIMENT_ID) -> bool:
    """BCI Main.

    The BCI main function will initialize a save folder, construct needed information
    and execute the task. This is the main connection between any UI and
    running the app.

    It may also be invoked via tha command line.
        Ex. `python bci_main.py` this will default parameters, mode, user, and type.

        You can pass it those attributes with flags, if desired.
            Ex. `python bci_main.py --user "bci_user" --task "RSVP Calibration"

    Input:
        parameter_location (str): location of parameters file to use
        user (str): name of the user
        task (TaskType): registered bcipy TaskType
        experiment_id (str): Name of the experiment. Default name is DEFAULT_EXPERIMENT_ID.


    """
    # Load parameters
    parameters = load_json_parameters(parameter_location, value_cast=True)

    # Update property to reflect the parameter source
    parameters['parameter_location'] = parameter_location
    if parameter_location != DEFAULT_PARAMETERS_PATH:
        parameters.save()
        default_params = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
        if parameters.add_missing_items(default_params):
            raise Exception('Parameters file out of date.')

    # update our parameters file with system related information
    sys_info = get_system_info()

    # Initialize Save Folder
    save_folder = init_save_data_structure(
        parameters['data_save_loc'],
        user,
        parameter_location,
        task=task.label,
        experiment_id=experiment)

    # configure bcipy session logging
    configure_logger(save_folder,
                     log_name=parameters['log_name'],
                     version=sys_info['bcipy_version'])

    logging.getLogger(__name__).info(sys_info)

    return execute_task(task, parameters, save_folder)


def execute_task(task: TaskType, parameters: dict, save_folder: str) -> bool:
    """Execute Task.

    Executes the desired task by setting up the display window and
        data acquisition, then passing on to the start_task function
        which will initialize experiment.

    Input:
        task(str): registered bcipy TaskType
        parameters (dict): parameter dictionary
        save_folder (str): path to save folder
    """
    signal_model = None
    language_model = None
    filename = None

    fake = parameters['fake_data']

    # Init EEG Model, if needed. Calibration Tasks Don't require probabilistic
    # modules to be loaded.
    if not fake and task not in TaskType.calibration_tasks():
        # Try loading in our signal_model and starting a langmodel(if enabled)
        try:
            signal_model, filename = load_signal_model()
        except Exception as e:
            print(f'Cannot load signal model. Exiting. {e}')
            raise e

        # if Language Model enabled init lm
        if parameters['lang_model_enabled']:
            language_model = init_language_model(parameters)

    # Initialize DAQ
    daq, server = init_eeg_acquisition(
        parameters, save_folder, server=fake)

    # Initialize Display Window
    # We have to wait until after the prompt to load the signal model before
    # displaying the window, otherwise in fullscreen mode this throws an error
    display = init_display_window(parameters)
    print_message(display, 'Initializing...')

    # Start Task
    try:
        start_task(
            display, daq, task, parameters, save_folder,
            language_model=language_model,
            signal_model=signal_model, fake=fake, auc_filename=filename)

    # If exception, close all display and acquisition objects
    except Exception as e:
        _clean_up_session(display, daq, server)
        raise e

    return _clean_up_session(display, daq, server)


def _clean_up_session(display, daq, server):
    """Clean up session."""

    # Stop Acquisition
    daq.stop_acquisition()
    daq.cleanup()

    if server:
        server.stop()

    # Close the display window
    # NOTE: There is currently a bug in psychopy when attempting to shutdown
    # windows when using a USB-C monitor. Putting the display close last in
    # the sequence allows acquisition to properly shutdown.
    display.close()

    return True


if __name__ == "__main__":
    import argparse
    import multiprocessing
    from bcipy.helpers.load import load_json_parameters

    # Needed for windows machines
    multiprocessing.freeze_support()

    task_options = TaskType.list()
    parser = argparse.ArgumentParser()

    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p', '--parameters', default=DEFAULT_PARAMETERS_PATH,
                        help='Parameter location. Pass as *.json')
    parser.add_argument('-u', '--user', default='test_user')
    parser.add_argument('-t', '--task', default='RSVP Calibration',
                        help=f'Task type to execute. Registered options: {task_options}')
    parser.add_argument('-e', '--experiment', default=DEFAULT_EXPERIMENT_ID, 
                        help='Select a valid experiment to run the task for this user')
    args = parser.parse_args()

    # Start BCI Main
    bci_main(args.parameters, str(args.user), TaskType.by_value(str(args.task)), str(args.experiment))
