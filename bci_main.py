import logging
import os
from bcipy.display.display_main import init_display_window
from bcipy.helpers.acquisition_related import init_eeg_acquisition
from bcipy.helpers.bci_task_related import print_message
from bcipy.helpers.lang_model_related import init_language_model
from bcipy.helpers.load import load_signal_model
from bcipy.helpers.save import init_save_data_structure
from bcipy.tasks.start_task import start_task
from bcipy.tasks.task_registry import ExperimentType



def bci_main(parameters: dict, user: str, exp_type: int, mode: str) -> bool:
    """BCI Main.

    The BCI main function will initialize a save folder, construct needed information
    and execute the task. This is the main connection between any UI and
    running the app.

    It may also be invoked via tha command line.
        Ex. `python bci_main.py` this will default parameters, mode, user, and type.

        You can pass it those attributes with flags, if desired.
            Ex. `python bci_main.py --user "bci_user" --mode "SHUFFLE"`

    Input:
        parameters (dict): parameter dictionary
        user (str): name of the user
        exp_type (int): type of experiment. Ex. 1 = calibration
        mode (str): BCI mode. Ex. RSVP, SHUFFLE, MATRIX

    """

    # Define the parameter and data save location
    parameter_location = parameters['parameter_location']
    data_save_location = parameters['data_save_loc']

    # Initialize Save Folder
    save_folder = init_save_data_structure(
        data_save_location, user, parameter_location, mode, exp_type)

    # Register Task Type
    task_type = {
        'mode': mode,
        'exp_type': exp_type
    }

    logfile = os.path.join(save_folder, 'logs', 'bcipy_session.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s',
        filename=logfile)
    print(f"Logging output to {logfile}")

    return execute_task(task_type, parameters, save_folder)


def execute_task(task_type: dict, parameters: dict, save_folder: str) -> bool:
    """Excecute Task.

    Executes the desired task by setting up the display window and
        data acquisition, then passing on to the start_task funtion
        which will initialize experiment.

    Input:
        parameters (dict): parameter dictionary
        task_type (dict): type and mode of experiment
        save_folder (str): path to save folder
    """

    exp_type = ExperimentType(task_type['exp_type'])

    fake = parameters['fake_data']

    signal_model = None
    language_model = None
    filename = None

    # Init EEG Model, if needed. Calibration Tasks Don't require probabilistic
    # modules to be loaded.
    if not fake and exp_type not in ExperimentType.calibration_tasks():
        # Try loading in our signal_model and starting a langmodel(if enabled)
        try:
            signal_model, filename = load_signal_model()
        except Exception as e:
            logging.debug('Cannot load signal model. Exiting')
            raise e

        # if Language Model enabled init lm
        if parameters['languagemodelenabled']:
            language_model = init_language_model(parameters)

    # Initialize Display Window
    # We have to wait until after the prompt to load the signal model before
    # displaying the window, otherwise in fullscreen mode this throws an error
    display = init_display_window(parameters)
    print_message(display, "Initializing...")

    # Initialize DAQ
    daq, server = init_eeg_acquisition(
        parameters, save_folder, server=fake)

    # Start Task
    try:
        start_task(
            display, daq, exp_type, parameters, save_folder,
            language_model=language_model,
            signal_model=signal_model, fake=fake, auc_filename=filename)

    # If exception, close all display and acquisition objects
    except Exception as e:
        _clean_up_session(display, daq, server)
        raise e

    return _clean_up_session(display, daq, server)


def _clean_up_session(display, daq, server):
    """Clean up session."""
    # Close the display window
    display.close()

    # Stop Acquisition
    daq.stop_acquisition()
    daq.cleanup()

    if server:
        server.stop()

    return True


if __name__ == "__main__":
    import argparse
    import multiprocessing
    from bcipy.helpers.load import load_json_parameters

    # Needed for windows machines
    multiprocessing.freeze_support()

    task_options = '; '.join([(f"{task.name.title().replace('_',' ')}:"
                               f" {task.value}")
                              for task in ExperimentType])
    parser = argparse.ArgumentParser()
    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p', '--parameters', default='bcipy/parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. Pass as parameters/parameters.json')
    parser.add_argument('-u', '--user', default='test_user')
    parser.add_argument('-t', '--type', default=1,
                        help=f'Task type. Options: ({task_options})')
    parser.add_argument('-m', '--mode', default='RSVP',
                        help='BCI mode. Ex. RSVP, MATRIX, SHUFFLE')
    args = parser.parse_args()

    # Load a parameters file
    parameters = load_json_parameters(args.parameters, value_cast=True)

    # Start BCI Main
    bci_main(parameters, str(args.user), int(args.type), str(args.mode))
