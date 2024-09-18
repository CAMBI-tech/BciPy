import unittest
import logging
from mockito import any, mock, when, unstub, verify
from bcipy.task.orchestrator import SessionOrchestrator
from bcipy.task import Task, TaskData
from bcipy.config import DEFAULT_PARAMETERS_PATH


class TestSessionOrchestrator(unittest.TestCase):
    parameter_location = DEFAULT_PARAMETERS_PATH

    def setUp(self) -> None:
        self.logger = mock(spec=logging.Logger)
        self.logger.info = lambda x: x
        self.logger.error = lambda x: x
        self.logger.exception = lambda x: x

    def tearDown(self) -> None:
        unstub()

    def test_orchestrator_add_task(self) -> None:
        task = mock(spec=Task)
        task.name = "test task"
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        orchestrator = SessionOrchestrator()
        self.assertTrue(orchestrator.tasks == [])
        orchestrator.add_task(task)
        self.assertTrue(len(orchestrator.tasks) == 1)

        verify(SessionOrchestrator, times=1)._init_orchestrator_save_folder(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_logger(any())

    def test_orchestrator_add_tasks(self) -> None:
        task1 = mock(spec=Task)
        task1.name = "test task"
        task2 = mock(spec=Task)
        task2.name = "test task"
        tasks = [task1, task2]
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        orchestrator = SessionOrchestrator()
        self.assertTrue(orchestrator.tasks == [])
        orchestrator.add_tasks(tasks)
        self.assertTrue(len(orchestrator.tasks) == 2)

        self.assertEqual(orchestrator.tasks[0], task1)
        self.assertEqual(orchestrator.tasks[1], task2)

        verify(SessionOrchestrator, times=1)._init_orchestrator_save_folder(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_logger(any())

    def test_orchestrator_execute(self) -> None:
        task = mock(spec=Task)
        task.name = "test task"
        task.execute = lambda: TaskData()
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator)._init_task_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_task_logger(any()).thenReturn()
        when(SessionOrchestrator)._save_protocol_data().thenReturn()
        when(task).__call__(
            any(),
            any(),
            fake=False,
            experiment_id=any(),
            parameters_path=any(),
            last_task_dir=None).thenReturn(task)
        when(self.logger).info(any()).thenReturn()
        orchestrator = SessionOrchestrator()
        orchestrator.add_task(task)
        orchestrator.execute()

        verify(task, times=1).__call__(
            any(), any(),
            fake=False, experiment_id=any(), parameters_path=any(), last_task_dir=None)
        verify(self.logger, times=1).info(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_save_folder(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_logger(any())
        verify(SessionOrchestrator, times=1)._init_task_save_folder(any())
        verify(SessionOrchestrator, times=1)._init_task_logger()
        verify(SessionOrchestrator, times=1)._save_protocol_data()


if __name__ == '__main__':
    unittest.main()
