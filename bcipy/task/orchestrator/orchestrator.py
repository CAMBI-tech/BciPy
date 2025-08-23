"""Task orchestration module for managing BCI experiment sessions.

This module provides functionality for managing and executing sequences of BCI tasks,
handling task initialization, execution, logging, and data management.
"""

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
from typing import Any, Dict, List, Optional, Tuple, Type

from bcipy.config import (DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_FILENAME,
                          DEFAULT_PARAMETERS_PATH, DEFAULT_USER_ID,
                          MULTIPHRASE_FILENAME, PROTOCOL_FILENAME,
                          PROTOCOL_LOG_FILENAME, SESSION_LOG_FILENAME)
from bcipy.core.parameters import Parameters
from bcipy.helpers.utils import configure_logger, get_system_info
from bcipy.io.load import load_json_parameters
from bcipy.task import Task, TaskData, TaskMode


class SessionOrchestrator:
    """Manages the execution of a protocol of BCI tasks.

    The Session Orchestrator is responsible for managing the execution of a sequence
    of tasks within an experiment session. It handles task initialization, execution
    order, data saving, and logging.

    Attributes:
        tasks: List of task classes to execute.
        task_names: List of task names in execution order.
        parameters: Configuration parameters for the session.
        sys_info: System information dictionary.
        log: Session logger instance.
        save_folder: Path where session data is saved.
        session_data: List of data from executed tasks.
        ready_to_execute: Whether tasks are ready to execute.
        last_task_dir: Path to the last executed task's directory.
        copyphrases: List of phrases for copy tasks.
        next_phrase: Next phrase to be used in copy tasks.
        starting_index: Starting index for copy tasks.
        user: User identifier.
        fake: Whether to use fake data.
        experiment_id: Experiment identifier.
        alert: Whether to alert when tasks complete.
        visualize: Whether to visualize task results.
        progress: Current task execution progress.
        user_exit: Whether user has requested to exit.
    """

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = DEFAULT_USER_ID,
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
        parameters: Optional[Parameters] = None,
        fake: bool = False,
        alert: bool = False,
        visualize: bool = False
    ) -> None:
        """Initialize the session orchestrator.

        Args:
            experiment_id: Identifier for the experiment session.
            user: User identifier.
            parameters_path: Path to parameters file.
            parameters: Optional pre-loaded parameters object.
            fake: Whether to use fake data for testing.
            alert: Whether to alert when tasks complete.
            visualize: Whether to visualize task results.
        """
        self.parameters_path = parameters_path
        if not parameters:
            self.parameters = load_json_parameters(
                parameters_path, value_cast=True)
        else:
            # This allows for the parameters to be passed in directly and modified before executions
            self.parameters = parameters

        self.copyphrases: Optional[List[Tuple[str, int]]] = None
        self.next_phrase: Optional[str] = None
        self.starting_index: int = 0

        self.initialize_copy_phrases()

        self.user = user
        self.fake = fake
        self.experiment_id = experiment_id
        self.sys_info = self.get_system_info()
        self.tasks: List[Type[Task]] = []
        self.task_names: List[str] = []
        self.session_data: List[TaskData] = []
        self.save_folder = self._init_orchestrator_save_folder(
            self.parameters["data_save_loc"])
        self.logger = self._init_orchestrator_logger(self.save_folder)

        self.alert = alert
        self.logger.info("Alerts are on") if self.alert else self.logger.info(
            "Alerts are off")
        self.visualize = visualize
        self.progress: int = 0

        self.ready_to_execute = False
        self.user_exit = False
        self.last_task_dir = None
        self.logger.info("Session Orchestrator initialized successfully")

    def add_task(self, task: Type[Task]) -> None:
        """Add a single task to the execution queue.

        Args:
            task: Task class to add to the queue.
        """
        self.tasks.append(task)
        self.task_names.append(task.name)
        self.ready_to_execute = True

    def add_tasks(self, tasks: List[Type[Task]]) -> None:
        """Add multiple tasks to the execution queue.

        Args:
            tasks: List of task classes to add to the queue.
        """
        for task in tasks:
            self.add_task(task)
        self.ready_to_execute = True

    def set_next_phrase(self) -> None:
        """Set the next phrase for copy phrase tasks.

        If there are phrases in the copyphrases list, uses the next one.
        Otherwise, uses the task_text from parameters.
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
        """Load copy phrases from a JSON file.

        The JSON file should be structured as:
        {
            "Phrases": [
                [string, int],
                [string, int],
                ...
            ]
        }

        If no file is provided, uses task_text from parameters.
        """
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
        """Execute all queued tasks in order.

        Raises:
            Exception: If no tasks have been added to the queue.
        """
        if not self.ready_to_execute:
            msg = "Orchestrator not ready to execute. No tasks have been added."
            self.logger.error(msg)
            raise Exception(msg)

        self.logger.info(
            f"Session Orchestrator executing tasks in order: {self.task_names}")
        for task in self.tasks:
            self.progress += 1
            if task.mode == TaskMode.COPYPHRASE:
                self.set_next_phrase()
            try:
                # initialize the task save folder and logger
                self.logger.info(
                    f"Initializing task {self.progress}/{len(self.tasks)} {task.name}")
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
                            self.logger.info(
                                f"Visualizing session data. Saving to {data_save_location}")
                            subprocess.run(
                                f'bcipy-erp-viz -s "{data_save_location}" '
                                f'--parameters "{self.parameters_path}" --show --save',
                                shell=True)
                        except Exception as e:
                            self.logger.info(
                                f'Error visualizing session data: {e}')

                initialized_task = None

            except Exception as e:
                self.logger.error(f"Task {task.name} failed to execute")
                self.logger.exception(e)
                try:
                    initialized_task.cleanup()  # type: ignore
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
        """Initialize the session logger.

        Args:
            save_folder: Directory to save log files.

        Returns:
            Logger: Configured logger instance.
        """
        return configure_logger(
            save_folder,
            PROTOCOL_LOG_FILENAME,
            logging.DEBUG)

    def _init_orchestrator_save_folder(self, save_path: str) -> str:
        """Initialize the session save directory.

        Args:
            save_path: Base path for saving session data.

        Returns:
            str: Path to the created save directory.
        """
        date_time = datetime.now()
        date = date_time.strftime("%Y-%m-%d")
        timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
        path = f'{save_path}{self.user}/{date}/{self.experiment_id}/{timestamp}/'
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'logs'), exist_ok=True)
        return path

    def _init_task_save_folder(self, task: Type[Task]) -> str:
        """Initialize a save directory for a task.

        Args:
            task: Task class to create directory for.

        Returns:
            str: Path to the created task directory.

        Raises:
            OSError: If directory creation fails for reasons other than
                the directory already existing.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = self.save_folder + \
            f'{task.name.replace(" ", "_")}_{timestamp}/'
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
            self.parameters.save(
                save_directory, name=DEFAULT_PARAMETERS_FILENAME)

        except OSError as error:
            # If the error is anything other than file existing, raise an error
            if error.errno != errno.EEXIST:
                raise error

        return save_directory

    def _init_task_logger(self, save_folder: str) -> None:
        """Initialize a logger for a task.

        Args:
            save_folder: Directory to save task logs.
        """
        configure_logger(save_folder, SESSION_LOG_FILENAME, logging.DEBUG)

    def _save_data(self) -> None:
        """Save all session data.

        Saves protocol data and copy phrases data to their respective files.
        """
        self._save_procotol_data()
        self._save_copy_phrases()

    def _save_procotol_data(self) -> None:
        """Save protocol data to a JSON file.

        Saves task names, system info, and other session metadata.
        """
        data = {
            'tasks': self.task_names,
            'sys_info': self.sys_info,
            'parameters': self.parameters_path,
            'user': self.user,
            'experiment_id': self.experiment_id,
            'fake': self.fake
        }
        with open(os.path.join(self.save_folder, PROTOCOL_FILENAME), 'w') as f:
            json.dump(data, f)

    def _save_copy_phrases(self) -> None:
        """Save copy phrases data to a JSON file.

        Only saves if copy phrases were used in the session.
        """
        if self.copyphrases:
            with open(os.path.join(self.save_folder, MULTIPHRASE_FILENAME), 'w') as f:
                json.dump({'Phrases': self.copyphrases}, f)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dict[str, Any]: Dictionary containing system information.
        """
        return get_system_info()

    def close_experiment_callback(self) -> None:
        """Callback for handling user-initiated experiment closure.

        Sets the user_exit flag to true to stop task execution.
        """
        self.user_exit = True
