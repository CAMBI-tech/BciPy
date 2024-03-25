from dataclasses import dataclass
import logging
from logging import Logger
from typing import List

from bcipy.helpers.exceptions import TaskConfigurationException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.validate import validate_bcipy_session, validate_experiment
from bcipy.helpers.visualization import visualize_session_data
from bcipy.main import execute_task
from bcipy.task import TaskType
from bcipy.config import DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH
from bcipy.signal.model import SignalModel
from bcipy.language.main import LanguageModel
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.save import init_save_data_structure
from bcipy.helpers.system_utils import configure_logger, get_system_info


@dataclass
class TaskInfo:
    """A Task to be executed, with config specific to it"""

    task_type: TaskType
    parameter_location: str
    visualize: bool = True
    alert: bool = False
    pause_after: bool = True


class SessionOrchestrator:
    tasks: List[TaskInfo]
    signal_models: List[SignalModel] = None
    language_model: LanguageModel = None
    parameters: Parameters
    sys_info: dict
    log: Logger
    save_folder: str = None
    # Session Orchestrator could contain global objects here (DAQ, models etc) to be shared between executed tasks.

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = "test_user",
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
    ) -> None:
        validate_experiment(experiment_id)
        self.parameters_path = parameters_path
        self.user = user
        self.experiment_id = experiment_id
        self.log = logging.getLogger(__name__)
        self.parameters = load_json_parameters(
            DEFAULT_PARAMETERS_PATH, value_cast=True
        )  # Refactor this to work with the more modular approach
        if not validate_bcipy_session(self.parameters, False):  # fake is false for now
            raise TaskConfigurationException("Invalid session parameters")
        self.parameters["parameter_location"] = parameters_path
        if parameters_path != DEFAULT_PARAMETERS_PATH:
            self.parameters.save()
            default_params = load_json_parameters(
                DEFAULT_PARAMETERS_PATH, value_cast=True
            )
            if self.parameters.add_missing_items(default_params):
                msg = "Parameters file out of date."
                self.log.exception(msg)
                raise Exception(msg)

        self.sys_info = get_system_info()
        self.tasks = []

    def queue_task(self, task: TaskInfo) -> None:
        # Loading task specific parameters could happen here
        self.tasks.append(task)
        if not self.save_folder:
            self.save_folder = init_save_data_structure(
                self.parameters["data_save_loc"],
                self.user,
                self.parameters_path,
                task=task.task_type.label,
                experiment_id=self.experiment_id,
            )
            configure_logger(self.save_folder, version=self.sys_info["bcipy_version"])

    def execute(self) -> None:
        """Executes queued tasks in order"""
        for task in self.tasks:
            if execute_task(
                task.task_type,
                self.parameters,
                self.save_folder,
                task.alert,
                fake=False,
            ):
                if task.visualize:
                    try:
                        visualize_session_data(self.save_folder, self.parameters)
                    except Exception as e:
                        self.log.info(f"Error visualizing session data: {e}")
            if task.pause_after:
                # pause here and show ui
                pass


def demo_orchestrator():
    orchestrator = SessionOrchestrator()
    task = TaskInfo(TaskType.RSVP_CALIBRATION, DEFAULT_PARAMETERS_PATH)
    orchestrator.queue_task(task)
    task = TaskInfo(TaskType.RSVP_COPY_PHRASE, DEFAULT_PARAMETERS_PATH)
    orchestrator.queue_task(task)
    orchestrator.execute()


if __name__ == "__main__":
    demo_orchestrator()
