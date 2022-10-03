import argparse
import logging
import multiprocessing

from typing import Optional

from psychopy import visual

from bcipy.acquisition import LslAcquisitionClient, LslDataServer
from bcipy.display import init_display_window
from bcipy.config import DEFAULT_PARAMETERS_PATH, DEFAULT_EXPERIMENT_ID, STATIC_AUDIO_PATH
from bcipy.helpers.acquisition import init_eeg_acquisition
from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import (load_experiments, load_json_parameters,
                                load_signal_model)
from bcipy.helpers.save import init_save_data_structure
from bcipy.helpers.session import collect_experiment_field_data
from bcipy.helpers.stimuli import play_sound
from bcipy.helpers.system_utils import configure_logger, get_system_info
from bcipy.helpers.task import print_message
from bcipy.helpers.validate import validate_experiment, validate_bcipy_session
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.task import TaskType
from bcipy.task.start_task import start_task

log = logging.getLogger(__name__)


def bci_main(
        parameter_location: str,
        user: str,
        task: TaskType,
        experiment: str = DEFAULT_EXPERIMENT_ID,
        alert: bool = False) -> bool:
    """BCI Main.

    The BCI main function will initialize a save folder, construct needed information
    and execute the task. This is the main connection between any UI and
    running the app.

    It may also be invoked via tha command line.
        Ex. `bcipy` this will default parameters, mode, user, and type.

        You can pass it those attributes with flags, if desired.
            Ex. `bcipy --user "bci_user" --task "RSVP Calibration" --experiment "default"

    Input:
        parameter_location (str): location of parameters file to use
        user (str): name of the user
        task (TaskType): registered bcipy TaskType
        experiment_id (str): Name of the experiment. Default name is DEFAULT_EXPERIMENT_ID.
        alert (bool): whether to alert the user when the task is complete
    """
    validate_experiment(experiment)
    # Load parameters
    parameters = load_json_parameters(parameter_location, value_cast=True)

    if not validate_bcipy_session(parameters):
        return False

    # Update property to reflect the parameter source
    parameters['parameter_location'] = parameter_location
    if parameter_location != DEFAULT_PARAMETERS_PATH:
        parameters.save()
        default_params = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
        if parameters.add_missing_items(default_params):
            msg = 'Parameters file out of date.'
            log.exception(msg)
            raise Exception(msg)

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
                     version=sys_info['bcipy_version'])

    log.info(sys_info)

    # Collect experiment field data
    collect_experiment_field_data(experiment, save_folder)

    return execute_task(task, parameters, save_folder, alert)


def execute_task(
        task: TaskType,
        parameters: dict,
        save_folder: str,
        alert: bool) -> bool:
    """Execute Task.

    Executes the desired task by setting up the display window and
        data acquisition, then passing on to the start_task function
        which will initialize experiment.

    Input:
        (str): registered bcipy TaskType
        parameters (dict): parameter dictionary
        save_folder (str): path to save folder
        alert (bool): whether to alert the user when the task is complete
    """
    signal_model = None
    language_model = None

    fake = parameters['fake_data']

    # Init EEG Model, if needed. Calibration Tasks Don't require probabilistic
    # modules to be loaded.
    if task not in TaskType.calibration_tasks():
        # Try loading in our signal_model and starting a langmodel(if enabled)
        if not fake:
            try:
                signal_model, _filename = load_signal_model(
                    model_class=PcaRdaKdeModel,
                    model_kwargs={'k_folds': parameters['k_folds']},
                    filename=parameters['signal_model_path'])
            except Exception as e:
                log.exception(f'Cannot load signal model. Exiting. {e}')
                raise e

        language_model = init_language_model(parameters)

    # Initialize DAQ and export the device configuration
    daq, server = init_eeg_acquisition(
        parameters, save_folder, export_spec=True, server=fake)

    # Initialize Display Window
    # We have to wait until after the prompt to load the signal model before
    # displaying the window, otherwise in fullscreen mode this throws an error
    display = init_display_window(parameters)
    print_message(display, f'Initializing {task}...')

    # Start Task
    try:
        start_task(
            display, daq, task, parameters, save_folder,
            language_model=language_model,
            signal_model=signal_model, fake=fake)

    # If exception, close all display and acquisition objects
    except Exception as e:
        log.exception(str(e))

    if alert:
        play_sound(f"{STATIC_AUDIO_PATH}/{parameters['alert_sound_file']}")

    return _clean_up_session(display, daq, server)


def _clean_up_session(
        display: visual.Window,
        daq: LslAcquisitionClient,
        server: Optional[LslDataServer] = None) -> bool:
    """Clean up session.

    Closes the display window and data acquisition objects. Returns True if the session was closed successfully.

    Input:
        display (visual.Window): display window
        daq (LslAcquisitionClient): data acquisition client
        server (LslDataServer): data server
    """
    try:
        # Stop Acquisition
        daq.stop_acquisition()
        daq.cleanup()

        if server:
            server.stop()

        # Close the display window
        # NOTE: There is currently a bug in psychopy when attempting to shutdown
        # windows when using a USB-C monitor. Putting the display close last in
        # the inquiry allows acquisition to properly shutdown.
        display.close()
        return True
    except Exception as e:
        log.exception(str(e))
        return False


def bcipy_main() -> None:
    """BciPy Main.

    Command line interface used for running a registered experiment task in Bcipy. To see what
        is available use the --help flag.
    """
    # Needed for windows machines
    multiprocessing.freeze_support()

    experiment_options = list(load_experiments().keys())
    task_options = TaskType.list()
    parser = argparse.ArgumentParser()

    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p', '--parameters', default=DEFAULT_PARAMETERS_PATH,
                        help='Parameter location. Pass as *.json')
    parser.add_argument('-u', '--user', default='test_user')
    parser.add_argument('-t', '--task', default='RSVP Calibration',
                        help=f'Task type to execute. Registered options: {task_options}')
    parser.add_argument(
        '-e',
        '--experiment',
        default=DEFAULT_EXPERIMENT_ID,
        help=f'Select a valid experiment to run the task for this user. Available options: {experiment_options}')
    parser.add_argument(
        '-a',
        '--alert',
        default=False,
        action='store_true',
        help='Alert the user when the task is complete.')
    args = parser.parse_args()

    # Start BCI Main
    bci_main(args.parameters, str(args.user), TaskType.by_value(str(args.task)), str(args.experiment), args.alert)


if __name__ == '__main__':
    bcipy_main()
