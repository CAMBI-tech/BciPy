from dataclasses import dataclass
import logging
from logging import Logger
from typing import List, Optional

from bcipy.helpers.exceptions import TaskConfigurationException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.validate import validate_bcipy_session, validate_experiment
from bcipy.helpers.visualization import visualize_session_data
from bcipy.main import execute_task
from bcipy.task import Task
from bcipy.config import DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH
from bcipy.signal.model import SignalModel
from bcipy.language.main import LanguageModel
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.save import init_save_data_structure
from bcipy.helpers.system_utils import get_system_info


from bcipy.orchestrator.actions import CodeHookAction


# Test SessionOrchestrator using Actions
# Should execute return a status code or a boolean?


class SessionOrchestrator:
    tasks: List[Task]
    signal_models: Optional[List[SignalModel]] = None
    language_model: LanguageModel = None
    parameters: Parameters
    sys_info: dict
    log: Logger
    save_folder: Optional[str] = None
    # Session Orchestrator could contain global objects here (DAQ, models etc) to be shared between executed tasks.

    def __init__(
        self,
        experiment_id: str = DEFAULT_EXPERIMENT_ID,
        user: str = "test_user",
        parameters_path: str = DEFAULT_PARAMETERS_PATH,
    ) -> None:
        validate_experiment(experiment_id)
        self.parameters_path = (
            parameters_path  # TODO: load parameters and cast them to the correct values
        )
        self.parameters = load_json_parameters(parameters_path, True)
        self.user = user
        self.experiment_id = experiment_id
        self.log = logging.getLogger(__name__)
        self.sys_info = get_system_info()
        self.tasks = []
        self.session_data = []
        # TODO create datasave structure and provide it to the tasks

        self.ready_to_execute = False

    def add_task(self, task) -> None:
        # Loading task specific parameters could happen here
        self.tasks.append(task)

    def execute(self) -> None:
        """Executes queued tasks in order"""
        for task in self.tasks:
            data_save_location = init_save_data_structure(
                self.parameters["data_save_loc"],
                'test_user', #TODO: Perhaps this should be a constant
                self.parameters_path,
                task=task.name,
                experiment_id=self.experiment_id,
            )
            self.session_data.append(data_save_location)
            task.setup(self.parameters, data_save_location)
            task.execute()
            task.cleanup()

    def save(self) -> None:
        # Save the session data
        ...
    