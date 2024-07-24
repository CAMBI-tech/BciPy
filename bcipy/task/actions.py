import subprocess
from typing import Callable, Optional

from bcipy.gui.experiments.ExperimentField import \
    start_experiment_field_collection_gui
from bcipy.task import Task
from bcipy.task.main import TaskData


class CallbackAction(Task):
    """
    Action for running a callback.
    """
    name = 'Callback Action'

    def __init__(self, callback: Callable, *args, **kwargs) -> None:
        super().__init__()
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def execute(self) -> TaskData:
        self.logger.info(f'Executing callback action {self.callback} with args {self.args} and kwargs {self.kwargs}')
        self.callback(*self.args, **self.kwargs)
        self.logger.info(f'Callback action {self.callback} executed')
        return TaskData()


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """
    name = 'Code Hook Action'

    def __init__(self, code_hook: str, subprocess=True) -> None:
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
    name = 'Offline Analysis Action'

    def __init__(self, data_directory: str, parameters_path: Optional[str] = None, alert=True) -> None:
        super().__init__()
        self.parameters_path = parameters_path
        self.alert = alert
        self.data_directory = data_directory
        self.command = self.construct_command()

    def execute(self) -> TaskData:
        cmd = self.construct_command()
        subprocess.Popen(cmd, shell=True)
        return TaskData(task_save=self.data_directory)

    def construct_command(self):
        command = 'python -m bcipy.signal.model.offline_analysis'
        command += ' --data_folder ' + self.data_directory
        if self.parameters_path:
            command += ' --parameters_file ' + self.parameters_path
        if self.alert:
            command += ' --alert'


class ExperimentFieldCollectionAction(Task):
    """
    Action for collecting experiment field data.
    """

    name = 'Experiment Field Collection Action'

    def __init__(self, experiment_id: str, save_path: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = save_path

    def execute(self) -> TaskData:
        self.logger.info(
            f'Collecting experiment field data for experiment {self.experiment_id} in save folder {self.save_folder}'
        )
        start_experiment_field_collection_gui(self.experiment_id,
                                              self.save_folder)
        return TaskData(task_save=self.save_folder)
