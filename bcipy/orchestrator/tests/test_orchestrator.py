import unittest
from unittest.mock import Mock
from bcipy.orchestrator.orchestrator import SessionOrchestrator
from bcipy.task import Task
from bcipy.task import TaskType
from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.helpers.exceptions import UnregisteredExperimentException


class TestSessionOrchestrator(unittest.TestCase):
    parameter_location = DEFAULT_PARAMETERS_PATH

    def test_orchestrator_invalid_experiment(self) -> None:
        experiment = "does not exist"
        with self.assertRaises(UnregisteredExperimentException):
            _ = SessionOrchestrator(experiment_id=experiment)

    def test_orchestrator_queues_task(self) -> None:
        task = Mock(spec=Task)
        orchestrator = SessionOrchestrator()
        assert len(orchestrator.tasks) == 0
        orchestrator.add_task(task)
        assert len(orchestrator.tasks) == 1
