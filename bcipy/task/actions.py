import subprocess
from typing import Callable, Optional

from bcipy.gui.experiments.ExperimentField import start_experiment_field_collection_gui
from bcipy.task import Task
from bcipy.task.main import TaskData
from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.signal.model.offline_analysis import offline_analysis


class CallbackAction(Task):
    """
    Action for running a callback.
    """

    name = "Callback Action"

    def __init__(self, callback: Optional[Callable] = None, *args, **kwargs) -> None:
        super().__init__()
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def execute(self) -> TaskData:
        self.logger.info(
            f"Executing callback action {self.callback} with args {self.args} and kwargs {self.kwargs}"
        )
        self.callback(*self.args, **self.kwargs)
        self.logger.info(f"Callback action {self.callback} executed")
        return TaskData()


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """

    name = "Code Hook Action"

    def __init__(self, parameters, data_directory, code_hook: Optional[str] = None, subprocess: bool=True, **kwargs) -> None:
        super().__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess

    def execute(self) -> TaskData:
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
        self, parameters, data_directory, alert=True, parameters_path: str = f'{DEFAULT_PARAMETERS_PATH}', last_task_dir: Optional[str] = None, **kwargs) -> None:
        super().__init__()
        self.parameters = parameters
        self.parameters_path = parameters_path
        self.alert = alert

        if last_task_dir:
            self.data_directory = last_task_dir
        else:
            self.data_directory = data_directory

    def execute(self) -> TaskData:
        response = offline_analysis(self.data_directory, self.parameters, alert_finished=self.alert)
        return TaskData(
            save_path=self.data_directory,
            task_dict={"parameters": self.parameters_path, "response": response},
        )

    def construct_command(self) -> str:
        command = "python -m bcipy.signal.model.offline_analysis"
        if self.parameters_path:
            command += " --parameters_file " + self.parameters_path
        if self.alert:
            command += " --alert"
        return command


class ExperimentFieldCollectionAction(Task):
    """
    Action for collecting experiment field data.
    """

    name = "Experiment Field Collection Action"

    def __init__(self, parameters, save_path, experiment_id: str = 'default', **kwargs) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = save_path
        self.parameters = parameters

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
