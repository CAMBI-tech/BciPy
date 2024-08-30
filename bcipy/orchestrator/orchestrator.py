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


class SessionOrchestrator:
    tasks: List[Task]
    task_names: List[str]
    parameters: Parameters
    sys_info: dict
    log: Logger
    save_folder: Optional[str] = None
    session_data: List[str]
    ready_to_execute: bool = False
    last_task_dir: Optional[str] = None

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = DEFAULT_USER_ID,
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
        fake: bool = False
    ) -> None:
        validate_experiment(experiment_id)
        self.parameters_path = parameters_path
        self.parameters = load_json_parameters(parameters_path, value_cast=True)
        self.user = user
        self.fake = fake
        self.experiment_id = experiment_id
        # TODO  load tasks from experiment protocol
        self.log = logging.getLogger(__name__)
        #TODO: fix logger, should be writing to a file per task and overall. See configure_logger.
        self.sys_info = get_system_info()
        self.tasks = []
        self.task_names = []
        self.session_data = []
        self.init_orchestrator_save_folder(self.parameters["data_save_loc"])

        self.ready_to_execute = True

    def add_task(self, task: Task) -> None:
        # Loading task specific parameters could happen here
        # TODO validate it is a Valid Task
        self.tasks.append(task)
        self.task_names.append(task.name)

    def execute(self) -> None:
        """Executes queued tasks in order"""

        for task in self.tasks:
            try:
                data_save_location = self.init_task_save_folder(task)
                self.session_data.append(data_save_location)
                initialized_task = task(self.parameters, data_save_location, fake=self.fake, experiment_id=self.experiment_id, parameters_path=self.parameters_path, last_task_dir=self.last_task_dir)
                initialized_task.execute()
                self.last_task_dir = data_save_location
            except Exception as e:
                self.log.exception(e)
        self.save()

    def init_orchestrator_save_folder(self, save_path: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # * No '/' after `save_folder` since it is included in
        # * `data_save_location` in parameters
        path = f'{save_path}{self.user}/{self.experiment_id}/{timestamp}/'
        os.makedirs(path)
        self.save_folder = path

    def init_task_save_folder(self, task: Task) -> str:
        assert self.save_folder is not None, "Orchestrator save folder not initialized"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = self.save_folder + f'{task.name}_{timestamp}/'
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
                'tasks': self.task_names,
                'parameters': self.parameters_path,
                'system_info': self.sys_info,
            }))
