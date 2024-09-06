import subprocess
from typing import Any, Optional
import logging

from bcipy.gui.experiments.ExperimentField import start_experiment_field_collection_gui
from bcipy.task import Task
from bcipy.helpers.parameters import Parameters
from bcipy.task.main import TaskData
from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.signal.model.offline_analysis import offline_analysis


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """

    name = "Code Hook Action"

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            logger: logging.Logger,
            code_hook: Optional[str] = None,
            subprocess: bool=True, **kwargs) -> None:
        super().__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess
        self.logger = logger

    def execute(self) -> TaskData:
        if self.code_hook:
            if self.subprocess:
                subprocess.Popen(self.code_hook, shell=True)

            else:
                subprocess.run(self.code_hook, shell=True)
        return TaskData()


class OfflineAnalysisAction(Task):
    """
    Action for running offline analysis.
    """

    name = "Offline Analysis Action"

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            logger: logging.Logger,
            alert=True,
            parameters_path: str = f'{DEFAULT_PARAMETERS_PATH}',
            last_task_dir: Optional[str] = None,
            **kwargs: Any) -> None:
        super().__init__()
        self.parameters = parameters
        self.parameters_path = parameters_path
        self.alert = alert
        self.logger = logger

        if last_task_dir:
            self.data_directory = last_task_dir
        else:
            self.data_directory = data_directory

    def execute(self) -> TaskData:
        response = offline_analysis(self.data_directory, self.parameters, alert_finished=self.alert)
        return TaskData(
            save_path=self.data_directory,
            task_dict={"parameters": self.parameters_path,
                       "response": response},
        )


class ExperimentFieldCollectionAction(Task):
    """
    Action for collecting experiment field data.
    """

    name = "Experiment Field Collection Action"

    def __init__(
            self,
            parameters: Parameters,
            save_path: str,
            logger: logging.Logger,
            experiment_id: str = 'default',
            **kwargs: Any) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = save_path
        self.parameters = parameters
        self.logger = logger

    def execute(self) -> TaskData:
        self.logger.info(
            f"Collecting experiment field data for experiment {self.experiment_id} in save folder {self.save_folder}"
        )
        start_experiment_field_collection_gui(self.experiment_id, self.save_folder)
        return TaskData(
            save_path=self.save_folder,
            task_dict={
                "experiment_id": self.experiment_id,
            },
        )
