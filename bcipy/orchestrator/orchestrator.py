import errno
import os
import json
from datetime import datetime
import logging
from logging import Logger
from typing import List, Optional, Union

from bcipy.helpers.parameters import Parameters
from bcipy.helpers.validate import validate_experiment
from bcipy.helpers.system_utils import get_system_info
from bcipy.task import Task
from bcipy.config import DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH, DEFAULT_USER_ID
from bcipy.signal.model import SignalModel
from bcipy.language.main import LanguageModel
from bcipy.helpers.load import load_json_parameters

"""
Session Orchestrator
--------------------

The Session Orchestrator is responsible for managing the execution of a protocol of tasks. It is initialized with an
experiment ID, user ID, and parameters file. Tasks are added to the orchestrator, which are then executed in order.
"""

# Session Orchestrator Needs:
# - A way to initialize the session (user, experiment, tasks, parameters, models, system info, log, save folder)
#   - save folder is not used in execute method and could be from a provided argument or from the parameters?
# - A way to save the session data


class SessionOrchestrator:
    tasks: List[Task]
    models: List[Union[SignalModel, LanguageModel]]
    parameters: Parameters
    sys_info: dict
    log: Logger
    save_folder: Optional[str] = None
    session_data: List[str]  # This may need to be a list of dictionaries or objects here in the future
    # Session Orchestrator will contain global objects here (DAQ, models etc) to be shared between executed tasks.

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = DEFAULT_USER_ID,
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
    ) -> None:
        validate_experiment(experiment_id)
        self.parameters_path = (
            parameters_path
        )
        self.parameters = load_json_parameters(parameters_path, True)
        self.user = user
        self.experiment_id = experiment_id
        self.log = logging.getLogger(__name__)
        self.sys_info = get_system_info()
        self.tasks = []
        self.session_data = []
        self.init_orchestrator_save_folder(self.parameters["data_save_loc"])

        self.ready_to_execute = False

    def add_task(self, task: Task) -> None:
        # Loading task specific parameters could happen here
        # TODO validate it is a Valid Task
        self.tasks.append(task)

    def execute(self) -> None:
        """Executes queued tasks in order"""

        # TODO add error handling for exceptions (like
        # TaskConfigurationException), allowing the orchestrator to continue and
        # log the errors.
        for task in self.tasks:
            data_save_location = self.init_task_save_folder(task)
            self.session_data.append(data_save_location)
            task.setup(self.parameters, data_save_location)
            task.execute()
            task.cleanup()
        self.save()

    # TODO: 'Runs' need a name like session or sequence.
    def init_orchestrator_save_folder(self, save_path: str) -> None:
        timestamp = str(datetime.now())
        # * No '/' after `save_folder` since it is included in
        # * `data_save_location` in parameters
        path = f'{save_path}{self.experiment_id}/{self.user}/orchestrator-run-{timestamp}/'
        os.makedirs(path)
        self.save_folder = path

    def init_task_save_folder(self, task: Task) -> str:
        assert self.save_folder is not None, "Orchestrator save folder not initialized"
        save_directory = self.save_folder + f'{self.user}_{task.name}/'
        try:
            # make a directory to save task data to
            os.makedirs(save_directory)
            os.makedirs(os.path.join(save_directory, 'logs'), exist_ok=True)
        except OSError as error:
            # If the error is anything other than file existing, raise an error
            if error.errno != errno.EEXIST:
                raise error
        return save_directory

    def save(self) -> None:
        # Save the session data
        system_info = get_system_info()
        with open(f'{self.save_folder}/session_data.json', 'w') as f:
            f.write(json.dumps({
                'system_info': self.sys_info,
            }))
