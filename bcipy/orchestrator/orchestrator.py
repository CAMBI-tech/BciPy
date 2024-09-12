import errno
import os
import json
from datetime import datetime
import logging
from logging import Logger
from typing import List, Type

from bcipy.helpers.parameters import Parameters
from bcipy.helpers.validate import validate_experiment
from bcipy.helpers.system_utils import get_system_info, configure_logger
from bcipy.task import Task, TaskData
from bcipy.config import DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH, DEFAULT_USER_ID
from bcipy.helpers.load import load_json_parameters

"""
Session Orchestrator
--------------------

The Session Orchestrator is responsible for managing the execution of a protocol of tasks. It is initialized with an
experiment ID, user ID, and parameters file. Tasks are added to the orchestrator, which are then executed in order.
"""


class SessionOrchestrator:
    tasks: List[Type[Task]]
    task_names: List[str]
    parameters: Parameters
    sys_info: dict
    log: Logger
    save_folder: str
    session_data: List[TaskData]
    ready_to_execute: bool = False
    last_task_dir: str

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
        self.sys_info = get_system_info()
        self.tasks = []
        self.task_names = []
        self.session_data = []
        self.init_orchestrator_save_folder(self.parameters["data_save_loc"])

        self.logger = configure_logger(
            self.save_folder,
            'protocol_log.txt',  # TODO: move to config
            logging.DEBUG,
            self.sys_info['bcipy_version'])

        self.ready_to_execute = True

    def add_task(self, task: Type[Task]) -> None:
        self.tasks.append(task)
        self.task_names.append(task.name)

    def execute(self) -> None:
        """Executes queued tasks in order"""

        if not self.ready_to_execute:
            msg = "Orchestrator not ready to execute tasks"
            self.log.error(msg)
            raise Exception(msg)

        for task in self.tasks:
            try:
                #  initialize the task save folder and logger
                data_save_location = self.init_task_save_folder(task)
                session_logger = configure_logger(
                    data_save_location,
                    log_level=logging.DEBUG,
                    version=self.sys_info['bcipy_version'])

                #  initialize the task and execute it
                initialized_task: Task = task(
                    self.parameters,
                    data_save_location,
                    session_logger,
                    fake=self.fake,
                    experiment_id=self.experiment_id,
                    parameters_path=self.parameters_path,
                    last_task_dir=self.last_task_dir)
                task_data = initialized_task.execute()
                self.session_data.append(task_data)
                self.logger.info(f"Task {task.name} completed successfully")
                # some tasks may need access to the previous task's data
                self.last_task_dir = data_save_location
            except Exception as e:
                self.logger.exception(e)
        self.save()

    def init_orchestrator_save_folder(self, save_path: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # * No '/' after `save_folder` since it is included in
        # * `data_save_location` in parameters
        path = f'{save_path}{self.user}/{self.experiment_id}/{timestamp}/'
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'logs'), exist_ok=True)
        self.save_folder = path

    def init_task_save_folder(self, task: Type[Task]) -> str:
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
        # Save the protocol data
        with open(f'{self.save_folder}/protocol.json', 'w') as f:
            f.write(json.dumps({
                'tasks': self.task_names,
                'parameters': self.parameters_path,
                'system_info': self.sys_info,
            }))
