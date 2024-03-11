from dataclasses import dataclass
import logging
from logging import Logger
from typing import List

from bcipy.task import TaskType
from bcipy.config import DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH
from bcipy.helpers.load import load_json_parameters, load_signal_models
from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.acquisition import init_eeg_acquisition
from bcipy.helpers.save import init_save_data_structure
from bcipy.helpers.system_utils import configure_logger, get_system_info


@dataclass
class TaskInfo:
    """A Task to be executed, with config specific to it"""

    task_type: TaskType
    parameter_location: str
    alert: bool = False
    pause_after: bool = True



class SessionOrchestrator:
    tasks: List[TaskInfo]
    # TODO: Import these
    signal_models: "SignalModel" = None
    language_model: "LanguageModel" = None
    
    sys_info: dict
    log: Logger

    # Session Orchestrator could contain global objects here (DAQ, models etc) to be shared between executed tasks.

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = "test_user",
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
    ) -> None:
        self.sys_info = get_system_info()

        self.user = user
        self.expiriment_id = experiment_id
        self.parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True) # Refactor this to work with the more modular approach
        self.log = logging.getLogger(__name__)
        # further configuration here once save folder handling is implemented
        self.tasks = []

    def queue_task(self, task: TaskInfo) -> None:
        # Loading task specific parameters could happen here
        self.tasks.append(task)

    def execute_task(self, task: TaskInfo) -> None:
        # Mostly copied from `main.py` for now. This will need to be refactored to be more Object Oriented.
        parameters = load_json_parameters(task.parameter_location, value_cast=True)
        if task.task_type not in TaskType.calibration_tasks():
            try:
                model_dir = parameters["signal_model_path"]
                self.signal_models = load_signal_models(directory=model_dir)
                assert self.signal_models, f"No signal models found in {model_dir}"
            except Exception as error:
                self.log.exception(f"Cannot load signal models. Exiting. {error}")

            # As this is only required for certain tasks, perhaps this should be functionality of task objects.

        #     if not self.language_model:
        #         init_language_model(parameters)
        # daq, servers = init_eeg_acquisition(parameters, save_folder)

        # daq, servers = init_eeg_acquisition(parameters, save)

    def execute(self) -> None:
        """Executes queued tasks in order"""
        for task in self.tasks:
            self.execute_task(task)
            if task.pause_after:
                # pause here and show ui
                pass


def test_orchestrator():
    pass


if __name__ == "__main__":
    test_orchestrator()
