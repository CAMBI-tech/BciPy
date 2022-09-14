import unittest
from mockito import any, unstub, when, mock, verify, verifyStubbedInvocationsAreUsed, verifyNoUnwantedInteractions

from bcipy import main
from bcipy.main import (
    bci_main,
    _clean_up_session,
    execute_task
)
from bcipy.config import DEFAULT_PARAMETERS_PATH, DEFAULT_EXPERIMENT_ID

from bcipy.helpers.exceptions import (
    UnregisteredExperimentException,
)
from bcipy.task import TaskType

import logging
logging.disable(logging.CRITICAL)


class TestBciMain(unittest.TestCase):

    parameter_location = DEFAULT_PARAMETERS_PATH
    data_save_location = '/'
    save_location = '/'
    parameters = {
        'data_save_loc': data_save_location,
        'log_name': 'test_log',
        'fake_data': False,
        'signal_model_path': ''
    }
    system_info = {
        'bcipy_version': 'test_version'
    }
    user = 'test_user'
    task = mock()
    task.label = 'RSVP Calibration'
    experiment = DEFAULT_EXPERIMENT_ID

    def tearDown(self) -> None:
        verifyStubbedInvocationsAreUsed()
        verifyNoUnwantedInteractions()
        unstub()

    def test_bci_main_default_experiment(self) -> None:
        mock_execute_response = mock()
        when(main).validate_experiment(self.experiment).thenReturn(True)
        when(main).validate_bcipy_session(self.parameters).thenReturn(True)
        when(main).load_json_parameters(self.parameter_location, value_cast=True).thenReturn(
            self.parameters
        )
        when(main).get_system_info().thenReturn(self.system_info)
        when(main).init_save_data_structure(
            self.data_save_location,
            self.user,
            self.parameter_location,
            task=self.task.label,
            experiment_id=self.experiment,
        ).thenReturn(self.save_location)
        when(main).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version']
        )
        when(main).collect_experiment_field_data(self.experiment, self.save_location)
        when(main).execute_task(self.task, self.parameters, self.save_location).thenReturn(mock_execute_response)

        response = bci_main(self.parameter_location, self.user, self.task)
        self.assertEqual(response, mock_execute_response)

        # validate all the calls happen as expected and the correct # of times
        verify(main, times=1).validate_experiment(self.experiment)
        verify(main, times=1).load_json_parameters(self.parameter_location, value_cast=True)
        verify(main, times=1).get_system_info()
        verify(main, times=1).init_save_data_structure(
            self.data_save_location,
            self.user,
            self.parameter_location,
            task=self.task.label,
            experiment_id=self.experiment)
        verify(main, times=1).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version'])
        verify(main, times=1).collect_experiment_field_data(self.experiment, self.save_location)
        verify(main, times=1).execute_task(self.task, self.parameters, self.save_location)

    def test_bci_main_invalid_experiment(self) -> None:
        experiment = 'does_not_exist'
        with self.assertRaises(UnregisteredExperimentException):
            bci_main(self.parameter_location, self.user, self.task, experiment)

    def test_invalid_parameter_location(self) -> None:
        invalid_paramater_location = 'does/not/exist.json'
        when(main).validate_experiment(self.experiment).thenReturn(True)

        with self.assertRaises(FileNotFoundError):
            bci_main(invalid_paramater_location, self.user, self.task)

        verify(main, times=1).validate_experiment(self.experiment)


class TestCleanUpSession(unittest.TestCase):

    def tearDown(self) -> None:
        unstub()

    def test_clean_up_no_server(self) -> None:
        daq = mock()
        display = mock()
        server = None

        # mock the required daq calls
        when(daq).stop_acquisition()
        when(daq).cleanup()

        # mock the required display call
        when(display).close()

        response = _clean_up_session(display, daq, server)
        self.assertTrue(response)

        verify(daq, times=1).stop_acquisition()
        verify(daq, times=1).cleanup()
        verify(display, times=1).close()

    def test_clean_up_with_server(self) -> None:
        daq = mock()
        display = mock()
        server = mock()

        # mock the required daq calls
        when(daq).stop_acquisition()
        when(daq).cleanup()

        # mock the required display call
        when(display).close()

        # mock the required server call
        when(server).stop()

        response = _clean_up_session(display, daq, server)
        self.assertTrue(response)

        verify(daq, times=1).stop_acquisition()
        verify(daq, times=1).cleanup()
        verify(display, times=1).close()
        verify(server, times=1).stop()


class TestExecuteTask(unittest.TestCase):

    def setUp(self) -> None:
        self.parameters = {
            'fake_data': True,
            'k_folds': 10,
            'is_txt_stim': True,
            'signal_model_path': ''
        }
        self.save_folder = '/'
        self.task = TaskType(1)
        self.display_mock = mock()
        self.daq = mock()
        self.server = mock()

    def tearDown(self) -> None:
        unstub()

    def test_execute_task_fake_data(self) -> None:
        response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True
        ).thenReturn(response)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_model=None,
            fake=self.parameters['fake_data'],
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_model=None,
            fake=self.parameters['fake_data'],
        )
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_task_real_data(self) -> None:
        self.parameters['fake_data'] = False
        response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True
        ).thenReturn(response)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_model=None,
            fake=self.parameters['fake_data'],
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_model=None,
            fake=self.parameters['fake_data'],
        )
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_task_non_calibration_real_data(self) -> None:
        self.parameters['fake_data'] = False
        model_path = "data/mycalib/"
        self.parameters['signal_model_path'] = model_path
        self.task = TaskType(2)
        signal_model = mock()
        language_model = mock()
        file_name = 'test'
        load_model_response = (signal_model, file_name)
        eeg_response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True
        ).thenReturn(eeg_response)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main).load_signal_model(model_class=any(), model_kwargs={
            'k_folds': self.parameters['k_folds']
        }, filename=model_path).thenReturn(load_model_response)
        when(main).init_language_model(self.parameters).thenReturn(language_model)
        when(main).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_model=signal_model,
            fake=self.parameters['fake_data'],
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_model=signal_model,
            fake=self.parameters['fake_data'],
        )
        verify(main, times=1).load_signal_model(model_class=any(), model_kwargs={
            'k_folds': self.parameters['k_folds']
        }, filename=model_path)
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_task_invalid_task(self) -> None:
        task = 'invalid'
        response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True
        ).thenReturn(response)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        with self.assertRaises(Exception):
            execute_task(task, self.parameters, self.save_folder)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_language_model_enabled(self) -> None:
        self.parameters['fake_data'] = False
        self.task = TaskType(2)  # set to a noncalibration task

        # mock the signal and language models
        signal_model = mock()
        file_name = 'test'
        language_model = mock()
        load_model_response = (signal_model, file_name)

        # mock the behavior of execute task
        eeg_response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True
        ).thenReturn(eeg_response)
        when(main).init_language_model(self.parameters).thenReturn(language_model)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main).load_signal_model(model_class=any(), model_kwargs={
            'k_folds': self.parameters['k_folds']
        }, filename='').thenReturn(load_model_response)
        when(main).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_model=signal_model,
            fake=self.parameters['fake_data'],
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.parameters['fake_data'],
            export_spec=True)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_model=signal_model,
            fake=self.parameters['fake_data'],
        )
        verify(main, times=1).load_signal_model(model_class=any(), model_kwargs={
            'k_folds': self.parameters['k_folds']
        }, filename='')
        verify(main, times=1).init_language_model(self.parameters)
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)


if __name__ == '__main__':
    unittest.main()
