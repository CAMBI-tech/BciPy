import argparse
import logging
import multiprocessing
from typing import Type

from bcipy.config import (DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH,
                          STATIC_AUDIO_PATH)
from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import (load_experiments, load_json_parameters,
                                load_signal_models)
from bcipy.helpers.save import init_save_data_structure
from bcipy.helpers.stimuli import play_sound
from bcipy.helpers.system_utils import configure_logger, get_system_info
from bcipy.helpers.validate import validate_bcipy_session, validate_experiment
from bcipy.helpers.visualization import visualize_session_data
from bcipy.task import TaskRegistry, Task
from bcipy.task.start import start_task

log = logging.getLogger(__name__)
task_registry = TaskRegistry()


def bci_main(
        parameter_location: str,
        user: str,
        task: Type[Task],
        experiment: str = DEFAULT_EXPERIMENT_ID,
        alert: bool = False,
        visualize: bool = True,
        fake: bool = False) -> bool:
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
        task (Task): registered bcipy Task
        experiment_id (str): Name of the experiment. Default name is DEFAULT_EXPERIMENT_ID.
        alert (bool): whether to alert the user when the task is complete
        visualize (bool): whether to visualize data at the end of a task
        fake (bool): whether to use fake acquisition data during the session. If None, the
            fake data will be determined by the parameters file.
    """
    validate_experiment(experiment)
    # Load parameters
    parameters = load_json_parameters(parameter_location, value_cast=True)

    # cli overrides parameters file for fake data if provided
    fake = fake if fake is True else parameters['fake_data']

    if not validate_bcipy_session(parameters, fake):
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
        task=task.name,
        experiment_id=experiment)

    # configure bcipy session logging
    configure_logger(save_folder,
                     version=sys_info['bcipy_version'])

    log.info(sys_info)

    # Collect experiment field data
    # collect_experiment_field_data(experiment, save_folder)

    if execute_task(task, parameters, save_folder, alert, fake):
        if visualize:

            # Visualize session data and fail silently if it errors
            try:
                visualize_session_data(save_folder, parameters)
            except Exception as e:
                log.info(f'Error visualizing session data: {e}')
        return True

    return False


def execute_task(
        task: Type[Task],
        parameters: dict,
        save_folder: str,
        alert: bool,
        fake: bool) -> bool:
    """Execute Task.

    Executes the desired task by setting up the display window and
        data acquisition, then passing on to the start_task function
        which will initialize experiment.

    Input:
        task(Task): Task that should be registered in TaskRegistry
        parameters (dict): parameter dictionary
        save_folder (str): path to save folder
        alert (bool): whether to alert the user when the task is complete
        fake (bool): whether to use fake acquisition data during the session

    Returns:
        (bool): True if the task was successfully executed, False otherwise
    """
    signal_models = []
    language_model = None

    # Init EEG Model, if needed. Calibration Tasks Don't require probabilistic
    # modules to be loaded.
    if task not in task_registry.calibration_tasks():
        # Try loading in our signal_model and starting a langmodel(if enabled)
        if not fake:
            try:
                model_dir = parameters['signal_model_path']
                signal_models = load_signal_models(directory=model_dir)
                assert signal_models, f"No signal models found in {model_dir}"
            except Exception as error:
                log.exception(f'Cannot load signal model. Exiting. {error}')
                raise error

        language_model = init_language_model(parameters)

    # Start Task
    try:
        start_task(task,
                   parameters,
                   save_folder,
                   language_model=language_model,
                   signal_models=signal_models,
                   fake=fake)

    # If exception, close all display and acquisition objects
    except Exception as e:
        log.exception(str(e))

    if alert:
        play_sound(f"{STATIC_AUDIO_PATH}/{parameters['alert_sound_file']}")

    return True


def bcipy_main() -> None:
    """BciPy Main.

    Command line interface used for running a registered experiment task in BciPy. To see what
        is available use the --help flag.
    """
    # Needed for windows machines
    multiprocessing.freeze_support()
    tr = TaskRegistry()
    experiment_options = list(load_experiments().keys())
    task_options = tr.list()
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
        help='Alert the user when the session is complete.')
    parser.add_argument(
        '-nv',
        '--noviz',
        default=True,
        action='store_false',
        help='Suppress visuals of the data after the session is complete.')
    parser.add_argument(
        '-f',
        '--fake',
        default=False,
        action='store_true',
        help='Use fake acquisition data for testing.')
    args = parser.parse_args()

    # Start BCI Main
    task = task_registry.get(args.task)
    bci_main(args.parameters, str(args.user), task,
             str(args.experiment), args.alert, args.noviz, args.fake)


if __name__ == '__main__':
    bcipy_main()
