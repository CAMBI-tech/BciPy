import subprocess
from typing import Any, Optional
import logging

from bcipy.gui.experiments.ExperimentField import start_experiment_field_collection_gui
from bcipy.task import Task, TaskData, TaskMode
from bcipy.helpers.parameters import Parameters
from bcipy.config import DEFAULT_PARAMETERS_PATH, SESSION_LOG_FILENAME
from bcipy.signal.model.offline_analysis import offline_analysis

logger = logging.getLogger(SESSION_LOG_FILENAME)


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """

    name = "CodeHookAction"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            code_hook: Optional[str] = None,
            subprocess: bool = True,
            **kwargs) -> None:
        super().__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess

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

    name = "OfflineAnalysisAction"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            parameters_path: str = f'{DEFAULT_PARAMETERS_PATH}',
            last_task_dir: Optional[str] = None,
            alert: bool = False,
            **kwargs: Any) -> None:
        super().__init__()
        self.parameters = parameters
        self.parameters_path = parameters_path
        self.alert_finished = alert

        if last_task_dir:
            self.data_directory = last_task_dir
        else:
            self.data_directory = data_directory

    def execute(self) -> TaskData:
        """Execute the offline analysis.

        Note: This function is called by the orchestrator to execute the offline analysis task. Some of the
            exceptions that can be raised by this function are not recoverable and will cause the orchestrator
            to stop execution. For example, if Exception is thrown in cross_validation due to the # of folds being
            inconsistent.

        """
        logger.info(f"Running offline analysis on data in {self.data_directory}")
        try:
            response = offline_analysis(self.data_directory, self.parameters, alert_finished=self.alert_finished)
        except Exception as e:
            logger.exception(f"Error running offline analysis: {e}")
            raise e
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
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            experiment_id: str = 'default',
            **kwargs: Any) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = data_directory
        self.parameters = parameters

    def execute(self) -> TaskData:
        logger.info(
            f"Collecting experiment field data for experiment {self.experiment_id} in save folder {self.save_folder}"
        )
        start_experiment_field_collection_gui(self.experiment_id, self.save_folder)
        return TaskData(
            save_path=self.save_folder,
            task_dict={
                "experiment_id": self.experiment_id,
            },
        )
