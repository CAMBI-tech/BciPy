import argparse
import logging
import multiprocessing
from typing import Optional, Type, List, Dict, Any, Union

from bcipy.config import CUSTOM_TASK_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH
from bcipy.exceptions import BciPyCoreException
from bcipy.helpers.validate import validate_bcipy_session, validate_experiment
from bcipy.io.load import load_experiments, load_json_parameters
from bcipy.task import Task, TaskRegistry
from bcipy.task.orchestrator import SessionOrchestrator
from bcipy.task.orchestrator.protocol import parse_protocol

logger = logging.getLogger(__name__)


def bci_main(
        parameter_location: str,
        user: str,
        experiment_id: Optional[str] = None,
        alert: bool = False,
        visualize: bool = True,
        fake: bool = False,
        task: Optional[Type[Task]] = None) -> bool:
    """Initialize and run a BCI task or experiment.

    The BCI main function initializes a save folder, constructs needed information
    and executes the task. This is the main connection between any UI and
    running the app.

    Args:
        parameter_location: Location of parameters file to use.
        user: Name of the user.
        experiment_id: Name of the experiment. If task is provided, this will be ignored.
        alert: Whether to alert the user when the task is complete.
        visualize: Whether to visualize data at the end of a task.
        fake: Whether to use fake acquisition data during the session. If None, the
            fake data will be determined by the parameters file.
        task: Registered bcipy Task to execute. If None, the task will be determined by the
            experiment protocol.

    Returns:
        bool: True if the task executed successfully, False otherwise.

    Raises:
        BciPyCoreException: If no experiment or task is provided.

    Examples:
        Command line usage:
            `bcipy` - uses default parameters, mode, user, and type
            `bcipy --user "bci_user" --task "RSVP Calibration"`
    """
    logger.info('Starting BciPy...')
    logger.info(
        f'User: {user} | Experiment: {experiment_id} | Task: {task} | '
        f'Parameters: {parameter_location} | '
        f'Alert: {alert} | Visualize: {visualize} | Fake: {fake}')
    # If no task is provided, extract the tasks from the experiment protocol. Otherwise, we will assume
    # the task is a custom task execution with no experiment attached.
    if not task and experiment_id:
        experiment = validate_experiment(experiment_id)
        # extract protocol from experiment
        tasks = parse_protocol(experiment['protocol'])
    elif task:
        tasks = [task]
        experiment_id = CUSTOM_TASK_EXPERIMENT_ID
    else:
        msg = 'No experiment or task provided to BciPy.'
        logger.exception(msg)
        raise BciPyCoreException(msg)

    # Load parameters
    parameters = load_json_parameters(parameter_location, value_cast=True)

    # cli overrides parameters file for fake data if provided
    fake = fake if fake is True else parameters['fake_data']
    parameters['fake_data'] = fake

    if not validate_bcipy_session(parameters, fake):
        return False

    # Update property to reflect the parameter source:
    parameters['parameter_location'] = parameter_location
    if parameter_location != DEFAULT_PARAMETERS_PATH:
        parameters.save()

    # Initialize an orchestrator
    orchestrator = SessionOrchestrator(
        experiment_id=experiment_id,
        user=user,
        parameters_path=parameter_location,
        parameters=parameters,
        fake=fake,
        alert=alert,
        visualize=visualize,
    )
    orchestrator.add_tasks(tasks)

    try:
        orchestrator.execute()
    except Exception as e:
        logger.exception(f'Error executing task: {e}')
        return False

    return True


def bcipy_main() -> None:
    """Command line interface for running BciPy experiments and tasks.

    This function provides a command line interface for running registered experiment 
    tasks in BciPy. It handles argument parsing and delegates execution to bci_main.

    Args:
        None

    Returns:
        None

    Note:
        Use the --help flag to see available options.
        Windows machines require multiprocessing support which is initialized here.
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
    parser.add_argument('-t', '--task', required=False,
                        help=f'Task type to execute. Registered options: {task_options}',
                        choices=task_options)
    parser.add_argument(
        '-e',
        '--experiment',
        required=False,
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

    if args.task:
        task = tr.get(args.task)
    else:
        task = None

    # Start BCI Main
    bci_main(
        args.parameters,
        str(args.user),
        str(args.experiment),
        args.alert,
        args.noviz,
        args.fake,
        task)


if __name__ == '__main__':
    bcipy_main()
