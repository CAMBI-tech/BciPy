# mypy: disable-error-code="arg-type, assignment"
import errno
import json
import logging
import os
import random
import subprocess
import time
from datetime import datetime
from logging import Logger
from typing import List, Optional, Type

from bcipy.config import (DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_FILENAME,
                          DEFAULT_PARAMETERS_PATH, DEFAULT_USER_ID,
                          MULTIPHRASE_FILENAME, PROTOCOL_FILENAME,
                          PROTOCOL_LOG_FILENAME, SESSION_LOG_FILENAME)
from bcipy.core.parameters import Parameters
from bcipy.helpers.utils import configure_logger, get_system_info
from bcipy.io.load import load_json_parameters
from bcipy.task import Task, TaskData, TaskMode


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

        self.copyphrases = None
        self.next_phrase = None
        self.starting_index = 0

        self.initialize_copy_phrases()

        self.user = user
        self.fake = fake
        self.experiment_id = experiment_id
        self.sys_info = self.get_system_info()
        self.tasks = []
        self.task_names = []
        self.session_data = []
        self.save_folder = self._init_orchestrator_save_folder(self.parameters["data_save_loc"])
        self.logger = self._init_orchestrator_logger(self.save_folder)

        self.alert = alert
        self.logger.info("Alerts are on") if self.alert else self.logger.info("Alerts are off")
        self.visualize = visualize
        self.progress = 0

        self.ready_to_execute = False
        self.user_exit = False
        self.logger.info("Session Orchestrator initialized successfully")

    def add_task(self, task: Type[Task]) -> None:
        """Add a task to the orchestrator"""
        self.tasks.append(task)
        self.task_names.append(task.name)
        self.ready_to_execute = True

    def add_tasks(self, tasks: List[Type[Task]]) -> None:
        """Add a list of tasks to the orchestrator"""
        for task in tasks:
            self.add_task(task)
        self.ready_to_execute = True

    def set_next_phrase(self) -> None:
        """Set the next phrase to be copied from the list of copy phrases loaded or the parameters directly.

        If there are no more phrases to copy, the task text and spelled letters from parameters will be used.
        """
        if self.copyphrases:
            if len(self.copyphrases) > 0:
                text, index = self.copyphrases.pop(0)
                self.next_phrase = text
                self.starting_index = index
            else:
                self.next_phrase = self.parameters['task_text']
        self.parameters['task_text'] = self.next_phrase
        self.parameters['spelled_letters_count'] = self.starting_index

    def initialize_copy_phrases(self) -> None:
        """Load copy phrases from a json file or take the task text if no file is provided.

        Expects a json file structured as follows:
        {
            "Phrases": [
                [string, int],
                [string, int],
                ...
            ]
        }
        """
        # load copy phrases from json file or take the task text if no file is provided
        if self.parameters.get('copy_phrases_location'):
            with open(self.parameters['copy_phrases_location'], 'r') as f:
                copy_phrases = json.load(f)
            self.copyphrases = copy_phrases['Phrases']
            # randomize the order of the phrases
            random.shuffle(self.copyphrases)
        else:
            self.copyphrases = None
            self.next_phrase = self.parameters['task_text']
            self.starting_index = self.parameters['spelled_letters_count']

    def execute(self) -> None:
        """Executes queued tasks in order"""

        if not self.ready_to_execute:
            msg = "Orchestrator not ready to execute. No tasks have been added."
            self.log.error(msg)
            raise Exception(msg)

        self.logger.info(f"Session Orchestrator executing tasks in order: {self.task_names}")
        for task in self.tasks:
            self.progress += 1
            if task.mode == TaskMode.COPYPHRASE:
                self.set_next_phrase()
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
                    alert_finished=self.alert,
                    experiment_id=self.experiment_id,
                    parameters_path=self.parameters_path,
                    protocol_path=self.save_folder,
                    last_task_dir=self.last_task_dir,
                    progress=self.progress,
                    tasks=self.tasks,
                    exit_callback=self.close_experiment_callback)
                task_data = initialized_task.execute()
                self.session_data.append(task_data)
                self.logger.info(f"Task {task.name} completed successfully")
                # some tasks may need access to the previous task's data
                self.last_task_dir = data_save_location

                if self.user_exit:
                    break

                if initialized_task.mode != TaskMode.ACTION:
                    if self.alert:
                        initialized_task.alert()

                    if self.visualize:
                        # Visualize session data and fail silently if it errors
                        try:
                            self.logger.info(f"Visualizing session data. Saving to {data_save_location}")
                            subprocess.run(
                                f'bcipy-erp-viz -s "{data_save_location}" '
                                f'--parameters "{self.parameters_path}" --show --save',
                                shell=True)
                        except Exception as e:
                            self.logger.info(f'Error visualizing session data: {e}')

                initialized_task = None

            except Exception as e:
                self.logger.error(f"Task {task.name} failed to execute")
                self.logger.exception(e)
                try:
                    initialized_task.cleanup()
                except BaseException:
                    pass

            # give the orchestrator time to save data before exiting
            time.sleep(1)

        # Save the protocol data and reset the orchestrator
        self._save_data()
        self.ready_to_execute = False
        self.tasks = []
        self.task_names = []
        self.progress = 0

    def _init_orchestrator_logger(self, save_folder: str) -> Logger:
        return configure_logger(
            save_folder,
            PROTOCOL_LOG_FILENAME,
            logging.DEBUG)

    def _init_orchestrator_save_folder(self, save_path: str) -> str:
        date_time = datetime.now()
        date = date_time.strftime("%Y-%m-%d")
        timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
        # * No '/' after `save_folder` since it is included in
        # * `data_save_location` in parameters
        path = f'{save_path}{self.user}/{date}/{self.experiment_id}/{timestamp}/'
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'logs'), exist_ok=True)
        return path

    def _init_task_save_folder(self, task: Type[Task]) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = self.save_folder + f'{task.name.replace(" ", "_")}_{timestamp}/'
        try:
            # make a directory to save task data to
            os.makedirs(save_directory)
            os.makedirs(os.path.join(save_directory, 'logs'), exist_ok=True)
            # save parameters to save directory with task name
            self.parameters.add_entry(
                "task",
                {
                    "value": task.name,
                    "section": "task_congig",
                    "name": "BciPy Task",
                    "helpTip": "A string representing the task that was executed",
                    "recommended": "",
                    "editable": "false",
                    "type": "str",
                }
            )
            self.parameters.save(save_directory, name=DEFAULT_PARAMETERS_FILENAME)

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

    def _save_data(self) -> None:

        self._save_procotol_data()
        # Save the remaining phrase data to a json file to be used in the next session
        if self.copyphrases and len(self.copyphrases) > 0:
            self._save_copy_phrases()

    def _save_procotol_data(self) -> None:
        # Save the protocol data to a json file
        with open(f'{self.save_folder}/{PROTOCOL_FILENAME}', 'w') as f:
            f.write(json.dumps({
                'tasks': self.task_names,
                'parameters': self.parameters_path,
                'system_info': self.sys_info,
            }))
            self.logger.info("Protocol data successfully saved")

    def _save_copy_phrases(self) -> None:
        # Save the copy phrases data to a json file
        with open(f'{self.save_folder}/{MULTIPHRASE_FILENAME}', 'w') as f:
            f.write(json.dumps({
                'Phrases': self.copyphrases
            }))
            self.logger.info("Copy phrases data successfully saved")

    def get_system_info(self) -> dict:
        return get_system_info()

    def close_experiment_callback(self):
        """Callback to close the experiment."""
        self.logger.info("User has exited the experiment.")
        self.user_exit = True
