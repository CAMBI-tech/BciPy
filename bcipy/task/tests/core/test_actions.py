import unittest
import subprocess

from mockito import mock, when, verify, unstub
from bcipy.task import actions, TaskData
from bcipy.task.actions import CodeHookAction, OfflineAnalysisAction, ExperimentFieldCollectionAction


class TestActions(unittest.TestCase):

    def setUp(self) -> None:
        self.parameters = mock()
        self.parameters_path = 'parameters_path'
        self.data_directory = 'data/'

    def tearDown(self) -> None:
        unstub()

    def test_code_hook_action_subprocess(self) -> None:
        code_hook = 'code_hook'
        when(subprocess).Popen(code_hook, shell=True).thenReturn(None)
        action = CodeHookAction(
            parameters=self.parameters,
            data_directory=self.data_directory,
            code_hook=code_hook,
            subprocess=True
        )
        response = action.execute()
        self.assertIsInstance(response, TaskData)
        verify(subprocess, times=1).Popen(code_hook, shell=True)

    def test_code_hook_action_no_subprocess(self) -> None:
        code_hook = 'code_hook'
        when(subprocess).run(code_hook, shell=True).thenReturn(None)
        action = CodeHookAction(
            parameters=self.parameters,
            data_directory=self.data_directory,
            code_hook=code_hook,
            subprocess=False
        )
        response = action.execute()
        self.assertIsInstance(response, TaskData)
        verify(subprocess, times=1).run(code_hook, shell=True)

    def test_offline_analysis_action(self) -> None:
        cmd_expected = f"bcipy-train --parameters '{self.parameters_path}'"

        when(subprocess).run(cmd_expected, shell=True, check=True).thenReturn(None)
        action = OfflineAnalysisAction(
            parameters=self.parameters,
            data_directory=self.data_directory,
            parameters_path=self.parameters_path,
        )
        response = action.execute()
        cmd_expected = f"bcipy-train --parameters '{self.parameters_path}'"
        self.assertIsInstance(response, TaskData)
        verify(subprocess, times=1).run(cmd_expected, shell=True, check=True)

    def test_experiment_field_collection_action(self) -> None:
        experiment_id = 'experiment_id'
        when(actions).start_experiment_field_collection_gui(experiment_id, self.data_directory).thenReturn(None)
        action = ExperimentFieldCollectionAction(
            parameters=self.parameters,
            data_directory=self.data_directory,
            experiment_id=experiment_id
        )
        task_data = action.execute()
        self.assertIsNotNone(task_data)
        self.assertIsInstance(task_data, TaskData)
        verify(actions, times=1).start_experiment_field_collection_gui(experiment_id, self.data_directory)


if __name__ == '__main__':
    unittest.main()
