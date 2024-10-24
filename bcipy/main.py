import argparse
import logging
import multiprocessing
from typing import Optional, Type

from bcipy.config import CUSTOM_TASK_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH
from bcipy.exceptions import BciPyCoreException
from bcipy.helpers.load import load_experiments, load_json_parameters
from bcipy.helpers.validate import validate_bcipy_session, validate_experiment
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
    """BCI Main.

    The BCI main function will initialize a save folder, construct needed information
    and execute the task. This is the main connection between any UI and
    running the app.

    A Task or Experiment ID must be provided to run the task. If a task is provided, the experiment
    ID will be ignored.

    It may also be invoked via tha command line.
        Ex. `bcipy` this will default parameters, mode, user, and type.

        You can pass it those attributes with flags, if desired.
            Ex. `bcipy --user "bci_user" --task "RSVP Calibration"


    Input:
        parameter_location (str): location of parameters file to use
        user (str): name of the user
        experiment_id (str): Name of the experiment. If task is provided, this will be ignored.
        alert (bool): whether to alert the user when the task is complete
        visualize (bool): whether to visualize data at the end of a task
        fake (bool): whether to use fake acquisition data during the session. If None, the
            fake data will be determined by the parameters file.
        task (Task): registered bcipy Task to execute. If None, the task will be determined by the
            experiment protocol.
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


def bcipy_main() -> None:  # pragma: no cover
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
