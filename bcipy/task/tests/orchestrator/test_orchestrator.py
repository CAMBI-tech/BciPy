import unittest
import logging
import json
from mock import mock_open
from mockito import any, mock, when, unstub, verify
from bcipy.task.orchestrator import SessionOrchestrator
from bcipy.task import Task, TaskData
from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.helpers.load import load_json_parameters


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
        task.mode = "test mode"
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
        task1.mode = "test mode"
        task2 = mock(spec=Task)
        task2.name = "test task"
        task2.mode = "test mode"
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
        task.mode = "test mode"
        task.execute = lambda: TaskData()
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator)._init_task_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_task_logger(any()).thenReturn()
        when(SessionOrchestrator)._save_data().thenReturn()
        when(task).__call__(
            any(),
            any(),
            fake=False,
            experiment_id=any(),
            alert_finished=any(),
            parameters_path=any(),
            last_task_dir=None,
            protocol_path=any(),
            progress=any(),
            tasks=any(),
            exit_callback=any(),
        ).thenReturn(task)
        orchestrator = SessionOrchestrator()
        orchestrator.add_task(task)
        orchestrator.execute()

        verify(task, times=1).__call__(
            any(),
            any(),
            fake=False,
            experiment_id=any(),
            alert_finished=any(),
            parameters_path=any(),
            last_task_dir=None,
            protocol_path=any(),
            progress=any(),
            tasks=any(),
            exit_callback=any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_save_folder(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_logger(any())
        verify(SessionOrchestrator, times=1)._init_task_save_folder(any())
        verify(SessionOrchestrator, times=1)._init_task_logger(any())
        verify(SessionOrchestrator, times=1)._save_data()

    @mock_open(read_data='{"Phrases": []}')
    def test_orchestrator_multiple_copyphrases_loads_from_parameters_when_set(self, mock_file):
        parameters = load_json_parameters(self.parameter_location, value_cast=True)
        copy_phrase_location = "bcipy/parameters/experiments/phrases.json"
        parameters['copy_phrases_location'] = copy_phrase_location
        mock_copy_phrases = {"Phrases": [["test", 0], ["test2", 1]]}
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator)._init_task_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_task_logger(any()).thenReturn()
        when(SessionOrchestrator)._save_data().thenReturn()
        when(json).load(mock_file).thenReturn(mock_copy_phrases)

        orchestrator = SessionOrchestrator(parameters=parameters)

        self.assertEqual(orchestrator.copyphrases, mock_copy_phrases['Phrases'])
        verify(json, times=1).load(mock_file)

    def test_orchestrator_save_data_multiple_copyphrases_saves_remaining_phrases(self):
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator)._init_task_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_task_logger(any()).thenReturn()
        when(SessionOrchestrator)._save_procotol_data().thenReturn()
        when(SessionOrchestrator)._save_copy_phrases().thenReturn()

        orchestrator = SessionOrchestrator()
        orchestrator.copyphrases = [["test", 0], ["test2", 1]]

        orchestrator._save_data()
        verify(SessionOrchestrator, times=1)._save_procotol_data()
        verify(SessionOrchestrator, times=1)._save_copy_phrases()

    def test_orchestrator_next_phrase(self):
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator)._init_task_save_folder(any()).thenReturn()
        when(SessionOrchestrator)._init_task_logger(any()).thenReturn()
        when(SessionOrchestrator)._save_procotol_data().thenReturn()
        when(SessionOrchestrator).initialize_copy_phrases().thenReturn()

        orchestrator = SessionOrchestrator()
        orchestrator.copyphrases = [["test", 5], ["test2", 1]]

        self.assertEqual(orchestrator.next_phrase, None)
        self.assertEqual(orchestrator.starting_index, 0)
        orchestrator.set_next_phrase()
        self.assertEqual(orchestrator.next_phrase, "test")
        self.assertEqual(orchestrator.starting_index, 5)
        self.assertTrue(len(orchestrator.copyphrases) == 1)


if __name__ == '__main__':
    unittest.main()
