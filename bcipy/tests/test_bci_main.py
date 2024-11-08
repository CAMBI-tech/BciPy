import logging
import unittest

from mockito import any, mock, unstub, verify, when

from bcipy import main
from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.exceptions import (BciPyCoreException,
                              UnregisteredExperimentException)
from bcipy.main import bci_main
from bcipy.task.orchestrator import SessionOrchestrator


class TestBCIMain(unittest.TestCase):

    def setUp(self) -> None:
        self.parameters_path = DEFAULT_PARAMETERS_PATH
        self.parameters = {
            'fake_data': False,
            'parameter_location': False,
            'visualize': False,
            'data_save_loc': 'data/',
        }
        self.user = 'test user'
        self.experiment = 'default'
        self.alert = False
        self.visualize = False
        self.fake = False
        self.logger = mock()
        self.logger.info = lambda x: x

    def tearDown(self) -> None:
        unstub()

    def test_bci_main_fails_without_experiment_or_task(self) -> None:
        with self.assertRaises(BciPyCoreException):
            bci_main(
                parameter_location=self.parameters_path,
                user=self.user,
                alert=self.alert,
                visualize=self.visualize,
                fake=self.fake
            )

    def test_bcipy_main_fails_with_invalid_experiment(self) -> None:
        when(main).validate_bcipy_session(any(), any()).thenRaise(UnregisteredExperimentException)
        with self.assertRaises(UnregisteredExperimentException):
            bci_main(
                parameter_location=self.parameters_path,
                user=self.user,
                experiment_id='invalid_experiment',
                alert=self.alert,
                visualize=self.visualize,
                fake=self.fake
            )

    def test_bci_main_runs_with_valid_experiment(self) -> None:
        when(main).validate_bcipy_session(any(), any()).thenReturn(True)  # Mock the validate_bcipy_session function
        when(main).load_json_parameters(
            any(), value_cast=any()).thenReturn(
            self.parameters)  # Mock the load_json_parameters function
        when(SessionOrchestrator).get_system_info().thenReturn(None)
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn(None)
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator).initialize_copy_phrases().thenReturn(None)
        when(SessionOrchestrator).add_tasks(any()).thenReturn(None)
        when(SessionOrchestrator).execute().thenReturn(None)

        bci_main(
            parameter_location=self.parameters_path,
            user=self.user,
            experiment_id=self.experiment,
            alert=self.alert,
            visualize=self.visualize,
            fake=self.fake
        )
        verify(SessionOrchestrator, times=1).add_tasks(any())
        verify(SessionOrchestrator, times=1).execute()
        verify(SessionOrchestrator, times=1).initialize_copy_phrases()
        verify(SessionOrchestrator, times=1)._init_orchestrator_logger(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_save_folder(any())
        verify(main, times=1).load_json_parameters(any(), value_cast=any())
        verify(SessionOrchestrator, times=1).get_system_info()

    def test_bci_main_runs_with_valid_task(self) -> None:
        when(main).validate_bcipy_session(any(), any()).thenReturn(True)
        when(main).load_json_parameters(any(), value_cast=any()).thenReturn(self.parameters)
        when(SessionOrchestrator).get_system_info().thenReturn(None)
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn(None)
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator).initialize_copy_phrases().thenReturn(None)
        when(SessionOrchestrator).add_tasks(any()).thenReturn(None)
        when(SessionOrchestrator).execute().thenReturn(None)

        bci_main(
            parameter_location=self.parameters_path,
            user=self.user,
            task='RSVP Calibration',
            alert=self.alert,
            visualize=self.visualize,
            fake=self.fake
        )

        verify(SessionOrchestrator, times=1).add_tasks(any())
        verify(SessionOrchestrator, times=1).execute()
        verify(SessionOrchestrator, times=1).initialize_copy_phrases()
        verify(SessionOrchestrator, times=1)._init_orchestrator_logger(any())
        verify(SessionOrchestrator, times=1)._init_orchestrator_save_folder(any())
        verify(main, times=1).load_json_parameters(any(), value_cast=any())
        verify(SessionOrchestrator, times=1).get_system_info()

    def test_bci_main_returns_false_with_orchestrator_execute_exception(self):
        when(main).validate_bcipy_session(any(), any()).thenReturn(True)
        when(main).load_json_parameters(any(), value_cast=any()).thenReturn(self.parameters)
        when(SessionOrchestrator).get_system_info().thenReturn(None)
        when(SessionOrchestrator)._init_orchestrator_save_folder(any()).thenReturn(None)
        when(SessionOrchestrator)._init_orchestrator_logger(any()).thenReturn(self.logger)
        when(SessionOrchestrator).initialize_copy_phrases().thenReturn(None)
        when(SessionOrchestrator).add_tasks(any()).thenReturn(None)
        when(SessionOrchestrator).execute().thenRaise(Exception)

        response = bci_main(
            parameter_location=self.parameters_path,
            user=self.user,
            task='RSVP Calibration',
            alert=self.alert,
            visualize=self.visualize,
            fake=self.fake
        )

        self.assertFalse(response)


if __name__ == '__main__':
    unittest.main()
