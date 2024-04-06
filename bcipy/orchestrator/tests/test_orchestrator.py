import unittest
from bcipy.orchestrator.session_orchestrator import SessionOrchestrator, TaskInfo
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
        orchestrator = SessionOrchestrator()
        task = TaskInfo(
            TaskType.RSVP_CALIBRATION,
            DEFAULT_PARAMETERS_PATH,
        )
        assert len(orchestrator.tasks) == 0
        orchestrator.queue_task(task)
        assert len(orchestrator.tasks) == 1
