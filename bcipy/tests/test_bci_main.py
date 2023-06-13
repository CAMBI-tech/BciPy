import logging
import unittest

from mockito import (any, mock, unstub, verify, verifyNoUnwantedInteractions,
                     verifyStubbedInvocationsAreUsed, when)

from bcipy import main
from bcipy.config import (DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH,
                          STATIC_AUDIO_PATH)
from bcipy.helpers.exceptions import UnregisteredExperimentException
from bcipy.main import _clean_up_session, bci_main, execute_task
from bcipy.task import TaskType

logging.disable(logging.CRITICAL)


class TestBciMain(unittest.TestCase):

    parameter_location = DEFAULT_PARAMETERS_PATH
    data_save_location = '/'
    save_location = '/'
    parameters = {
        'data_save_loc': data_save_location,
        'log_name': 'test_log',
        'fake_data': False,
        'signal_model_path': '',
        'lm_path': '',
        'alert_sound_file': 'test.wav',
    }
    system_info = {
        'bcipy_version': 'test_version'
    }
    user = 'test_user'
    task = mock()
    task.label = 'RSVP Calibration'
    experiment = DEFAULT_EXPERIMENT_ID
    alert = False
    fake = parameters['fake_data']

    def tearDown(self) -> None:
        verifyStubbedInvocationsAreUsed()
        verifyNoUnwantedInteractions()
        unstub()

    def test_bci_main_default_experiment(self) -> None:
        when(main).validate_experiment(self.experiment).thenReturn(True)
        when(main).validate_bcipy_session(self.parameters, self.fake).thenReturn(True)
        when(main).load_json_parameters(self.parameter_location, value_cast=True).thenReturn(
            self.parameters
        )
        when(main).visualize_session_data(self.save_location, self.parameters).thenReturn(None)
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
        when(main).execute_task(
            self.task,
            self.parameters,
            self.save_location,
            self.alert,
            self.fake).thenReturn(True)

        response = bci_main(self.parameter_location, self.user, self.task)
        self.assertEqual(response, True)

        # validate all the calls happen as expected and the correct # of times
        verify(main, times=1).validate_experiment(self.experiment)
        verify(main, times=1).validate_bcipy_session(self.parameters, self.fake)
        verify(main, times=1).load_json_parameters(self.parameter_location, value_cast=True)
        verify(main, times=1).get_system_info()
        verify(main, times=1).visualize_session_data(self.save_location, self.parameters)
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
        verify(main, times=1).execute_task(self.task, self.parameters, self.save_location, self.alert, self.fake)

    def test_bci_main_invalid_experiment(self) -> None:
        experiment = 'does_not_exist'
        with self.assertRaises(UnregisteredExperimentException):
            bci_main(self.parameter_location, self.user, self.task, experiment)

    def test_invalid_parameter_location(self) -> None:
        invalid_parameter_location = 'does/not/exist.json'
        when(main).validate_experiment(self.experiment).thenReturn(True)

        with self.assertRaises(FileNotFoundError):
            bci_main(invalid_parameter_location, self.user, self.task)

        verify(main, times=1).validate_experiment(self.experiment)

    def test_bci_main_visualize(self) -> None:
        """Test bci_main with visualization enabled."""
        when(main).validate_experiment(self.experiment).thenReturn(True)
        when(main).validate_bcipy_session(self.parameters, self.fake).thenReturn(True)
        when(main).load_json_parameters(self.parameter_location, value_cast=True).thenReturn(
            self.parameters
        )
        when(main).visualize_session_data(self.save_location, self.parameters).thenReturn(None)
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
        when(main).execute_task(
            self.task,
            self.parameters,
            self.save_location,
            self.alert,
            self.fake).thenReturn(True)

        response = bci_main(self.parameter_location, self.user, self.task, visualize=True)
        self.assertEqual(response, True)

        # validate all the calls happen as expected and the correct # of times
        verify(main, times=1).validate_experiment(self.experiment)
        verify(main, times=1).validate_bcipy_session(self.parameters, self.fake)
        verify(main, times=1).load_json_parameters(self.parameter_location, value_cast=True)
        verify(main, times=1).get_system_info()
        verify(main, times=1).visualize_session_data(self.save_location, self.parameters)
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
        verify(main, times=1).execute_task(self.task, self.parameters, self.save_location, self.alert, self.fake)

    def test_bci_main_visualize_disabled(self) -> None:
        """Test bci_main with visualization disabled."""
        when(main).validate_experiment(self.experiment).thenReturn(True)
        when(main).validate_bcipy_session(self.parameters, self.fake).thenReturn(True)
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
        when(main).execute_task(
            self.task,
            self.parameters,
            self.save_location,
            self.alert,
            self.fake).thenReturn(True)

        response = bci_main(self.parameter_location, self.user, self.task, visualize=False)
        self.assertEqual(response, True)

        # validate all the calls happen as expected and the correct # of times
        verify(main, times=1).validate_experiment(self.experiment)
        verify(main, times=1).validate_bcipy_session(self.parameters, self.fake)
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
        verify(main, times=1).execute_task(self.task, self.parameters, self.save_location, self.alert, self.fake)

    def test_bci_main_fake(self) -> None:
        """Test bci_main with fake data override."""
        fake = True
        when(main).validate_experiment(self.experiment).thenReturn(True)
        when(main).validate_bcipy_session(self.parameters, fake).thenReturn(True)
        when(main).load_json_parameters(self.parameter_location, value_cast=True).thenReturn(
            self.parameters
        )
        when(main).visualize_session_data(self.save_location, self.parameters).thenReturn(None)
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
        when(main).execute_task(
            self.task,
            self.parameters,
            self.save_location,
            self.alert,
            fake).thenReturn(True)

        response = bci_main(self.parameter_location, self.user, self.task, fake=fake)
        self.assertEqual(response, True)

        # validate all the calls happen as expected and the correct # of times
        verify(main, times=1).validate_experiment(self.experiment)
        verify(main, times=1).validate_bcipy_session(self.parameters, fake)
        verify(main, times=1).load_json_parameters(self.parameter_location, value_cast=True)
        verify(main, times=1).get_system_info()
        verify(main, times=1).visualize_session_data(self.save_location, self.parameters)
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
        verify(main, times=1).execute_task(self.task, self.parameters, self.save_location, self.alert, fake)


class TestCleanUpSession(unittest.TestCase):

    def tearDown(self) -> None:
        unstub()

    def test_clean_up_no_server(self) -> None:
        daq = mock()
        display = mock()
        servers = []

        # mock the required daq calls
        when(daq).stop_acquisition()
        when(daq).cleanup()

        # mock the required display call
        when(display).close()

        response = _clean_up_session(display, daq, servers)
        self.assertTrue(response)

        verify(daq, times=1).stop_acquisition()
        verify(daq, times=1).cleanup()
        verify(display, times=1).close()

    def test_clean_up_with_server(self) -> None:
        daq = mock()
        display = mock()
        server = mock()
        servers = [server]

        # mock the required daq calls
        when(daq).stop_acquisition()
        when(daq).cleanup()

        # mock the required display call
        when(display).close()

        # mock the required server call
        when(server).stop()

        response = _clean_up_session(display, daq, servers)
        self.assertTrue(response)

        verify(daq, times=1).stop_acquisition()
        verify(daq, times=1).cleanup()
        verify(display, times=1).close()
        verify(server, times=1).stop()


class TestExecuteTask(unittest.TestCase):

    def setUp(self) -> None:
        self.parameters = {
            'k_folds': 10,
            'is_txt_stim': True,
            'signal_model_path': '',
            'alert_sound_file': 'test.wav',
        }
        self.save_folder = '/'
        self.alert = False
        self.task = TaskType(1)
        self.fake = True
        self.display_mock = mock()
        self.daq = mock()
        self.eeg_client = mock()
        when(self.daq).get_client('EEG').thenReturn(self.eeg_client)
        self.server = [mock()]

    def tearDown(self) -> None:
        unstub()

    def test_execute_task_fake_data(self) -> None:
        response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake
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
            signal_models=[],
            fake=self.fake,
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder, self.alert, self.fake)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_task_real_data(self) -> None:
        self.fake = False
        response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake
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
            signal_models=[],
            fake=self.fake,
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder, self.alert, self.fake)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_task_non_calibration_real_data(self) -> None:
        self.fake = False
        model_path = "data/mycalib/"
        self.parameters['signal_model_path'] = model_path
        self.task = TaskType(2)
        signal_model = mock()
        language_model = mock()
        file_name = 'test'
        load_model_response = [signal_model]
        eeg_response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake).thenReturn(eeg_response)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main).load_signal_models(directory=model_path).thenReturn(load_model_response)
        when(main).init_language_model(self.parameters).thenReturn(language_model)
        when(main).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_models=[signal_model],
            fake=self.fake,
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder, self.alert, self.fake)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_models=[signal_model],
            fake=self.fake,
        )
        verify(main, times=1).load_signal_models(directory=model_path)
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_language_model_enabled(self) -> None:
        self.fake = False
        self.task = TaskType(2)  # set to a noncalibration task

        # mock the signal and language models
        signal_model = mock()
        file_name = 'test'
        language_model = mock()
        load_model_response = [signal_model]

        # mock the behavior of execute task
        eeg_response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake).thenReturn(eeg_response)
        when(main).init_language_model(self.parameters).thenReturn(language_model)
        when(main).init_display_window(self.parameters).thenReturn(self.display_mock)
        when(main).print_message(self.display_mock, any())
        when(main).load_signal_models(directory='').thenReturn(load_model_response)
        when(main).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_models=[signal_model],
            fake=self.fake,
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)

        execute_task(self.task, self.parameters, self.save_folder, self.alert, self.fake)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_models=[signal_model],
            fake=self.fake,
        )
        verify(main, times=1).load_signal_models(directory='')
        verify(main, times=1).init_language_model(self.parameters)
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)

    def test_execute_with_alert_enabled(self):
        expected_alert_path = f"{STATIC_AUDIO_PATH}/{self.parameters['alert_sound_file']}"
        response = (self.daq, self.server)
        when(main).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake,
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
            signal_models=[],
            fake=self.fake,
        )
        when(main)._clean_up_session(self.display_mock, self.daq, self.server)
        when(main).play_sound(expected_alert_path)

        execute_task(self.task, self.parameters, self.save_folder, True, self.fake)

        verify(main, times=1).init_eeg_acquisition(
            self.parameters,
            self.save_folder,
            server=self.fake)
        verify(main, times=1).init_display_window(self.parameters)
        verify(main, times=1).print_message(self.display_mock, any())
        verify(main, times=1).start_task(
            self.display_mock,
            self.daq,
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )
        verify(main, times=1)._clean_up_session(self.display_mock, self.daq, self.server)
        verify(main, times=1).play_sound(expected_alert_path)


if __name__ == '__main__':
    unittest.main()
