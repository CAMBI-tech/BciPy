import errno
import os
import json
from datetime import datetime
import logging
from logging import Logger
from typing import List, Type, Optional

from bcipy.helpers.parameters import Parameters
from bcipy.helpers.system_utils import get_system_info, configure_logger
from bcipy.task import Task, TaskData
from bcipy.config import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_PARAMETERS_PATH,
    DEFAULT_USER_ID,
    PROTOCOL_LOG_FILENAME,
    SESSION_LOG_FILENAME
)
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.visualization import visualize_session_data


class SessionOrchestrator:
    """
    Session Orchestrator
    --------------------

    The Session Orchestrator is responsible for managing the execution of a protocol of tasks. It is initialized with an
    experiment ID, user ID, and parameters file. Tasks are added to the orchestrator, which are then executed in order.
    """
    tasks: List[Type[Task]]
    task_names: List[str]
    parameters: Parameters
    sys_info: dict
    log: Logger
    save_folder: str
    session_data: List[TaskData]
    ready_to_execute: bool = False
    last_task_dir: Optional[str] = None

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = DEFAULT_USER_ID,
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
        parameters: Parameters = None,
        fake: bool = False,
        alert: bool = False,
        visualize: bool = False
    ) -> None:
        self.parameters_path = parameters_path
        if not parameters:
            self.parameters = load_json_parameters(parameters_path, value_cast=True)
        else:
            # This allows for the parameters to be passed in directly and modified before executions
            self.parameters = parameters

        self.user = user
        self.fake = fake
        self.experiment_id = experiment_id
        self.sys_info = get_system_info()
        self.tasks = []
        self.task_names = []
        self.session_data = []
        self.save_folder = self._init_orchestrator_save_folder(self.parameters["data_save_loc"])
        self.logger = self._init_orchestrator_logger(self.save_folder)

        self.alert = alert
        self.visualize = visualize
        self.progress = 0

        self.ready_to_execute = True
        self.logger.info("Session Orchestrator initialized successfully")

    def add_task(self, task: Type[Task]) -> None:
        self.tasks.append(task)
        self.task_names.append(task.name)

    def add_tasks(self, tasks: List[Type[Task]]) -> None:
        for task in tasks:
            self.add_task(task)

    def execute(self) -> None:
        """Executes queued tasks in order"""

        if not self.ready_to_execute:
            msg = "Orchestrator not ready to execute tasks"
            self.log.error(msg)
            raise Exception(msg)

        self.logger.info(f"Session Orchestrator executing tasks in order: {self.task_names}")
        for task in self.tasks:
            self.progress += 1
            try:
                # initialize the task save folder and logger
                self.logger.info(f"Initializing task {self.progress}/{len(self.tasks)} {task.name}")
                data_save_location = self._init_task_save_folder(task)
                self._init_task_logger(data_save_location)

                #  initialize the task and execute it
                initialized_task: Task = task(
                    self.parameters,
                    data_save_location,
                    fake=self.fake,
                    experiment_id=self.experiment_id,
                    parameters_path=self.parameters_path,
                    last_task_dir=self.last_task_dir)
                task_data = initialized_task.execute()
                self.session_data.append(task_data)
                self.logger.info(f"Task {task.name} completed successfully")
                # some tasks may need access to the previous task's data
                self.last_task_dir = data_save_location

                if self.alert:
                    initialized_task.alert()

                if self.visualize:
                    # Visualize session data and fail silently if it errors
                    try:
                        visualize_session_data(data_save_location, self.parameters)
                        pass
                    except Exception as e:
                        self.logger.info(f'Error visualizing session data: {e}')

            except Exception as e:
                self.logger.error(f"Task {task.name} failed to execute")
                self.logger.exception(e)

        self._save_protocol_data()

    def _init_orchestrator_logger(self, save_folder: str) -> Logger:
        return configure_logger(
            save_folder,
            PROTOCOL_LOG_FILENAME,
            logging.DEBUG)

    def _init_orchestrator_save_folder(self, save_path: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # * No '/' after `save_folder` since it is included in
        # * `data_save_location` in parameters
        path = f'{save_path}{self.user}/{self.experiment_id}/{timestamp}/'
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'logs'), exist_ok=True)
        return path

    def _init_task_save_folder(self, task: Type[Task]) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = self.save_folder + f'{task.name}_{timestamp}/'
        try:
            # make a directory to save task data to
            os.makedirs(save_directory)
            os.makedirs(os.path.join(save_directory, 'logs'), exist_ok=True)
            # save parameters to save directory
            self.parameters.save(save_directory)

        except OSError as error:
            # If the error is anything other than file existing, raise an error
            if error.errno != errno.EEXIST:
                raise error
        return save_directory

    def _init_task_logger(self, save_folder: str) -> None:
        configure_logger(
            save_folder,
            SESSION_LOG_FILENAME,
            logging.DEBUG)

    def _save_protocol_data(self) -> None:
        # Save the protocol data
        with open(f'{self.save_folder}/protocol.json', 'w') as f:  # TODO: move to config
            f.write(json.dumps({
                'tasks': self.task_names,
                'parameters': self.parameters_path,
                'system_info': self.sys_info,
            }))
            self.logger.info("Protocol data successfully saved")
