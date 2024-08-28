import logging
import unittest

from mockito import (any, mock, unstub, verify, verifyNoUnwantedInteractions,
                     verifyStubbedInvocationsAreUsed, when)

from bcipy import main
from bcipy.config import (DEFAULT_EXPERIMENT_ID, DEFAULT_PARAMETERS_PATH,
                          STATIC_AUDIO_PATH)
from bcipy.helpers.exceptions import UnregisteredExperimentException
from bcipy.main import bci_main, execute_task
from bcipy.task.paradigm.rsvp.calibration.calibration import RSVPCalibrationTask
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask

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
    task.name = 'RSVP Calibration'
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
            task=self.task.name,
            experiment_id=self.experiment,
        ).thenReturn(self.save_location)
        when(main).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version']
        )
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
            task=self.task.name,
            experiment_id=self.experiment)
        verify(main, times=1).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version'])
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
            task=self.task.name,
            experiment_id=self.experiment,
        ).thenReturn(self.save_location)
        when(main).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version']
        )
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
            task=self.task.name,
            experiment_id=self.experiment)
        verify(main, times=1).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version'])
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
            task=self.task.name,
            experiment_id=self.experiment,
        ).thenReturn(self.save_location)
        when(main).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version']
        )
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
            task=self.task.name,
            experiment_id=self.experiment)
        verify(main, times=1).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version'])
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
            task=self.task.name,
            experiment_id=self.experiment,
        ).thenReturn(self.save_location)
        when(main).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version']
        )
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
            task=self.task.name,
            experiment_id=self.experiment)
        verify(main, times=1).configure_logger(
            self.save_location,
            version=self.system_info['bcipy_version'])
        verify(main, times=1).execute_task(self.task, self.parameters, self.save_location, self.alert, fake)


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
        self.task = RSVPCalibrationTask
        self.fake = True
        self.display_mock = mock()
        self.server = [mock()]

    def tearDown(self) -> None:
        unstub()

    def test_execute_task_fake_data(self) -> None:

        when(main).start_task(
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )

        execute_task(self.task, self.parameters, self.save_folder, self.alert, self.fake)

        verify(main, times=1).start_task(
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )


    def test_execute_language_model_enabled(self) -> None:
        self.fake = False
        self.task = RSVPCopyPhraseTask  # set to a noncalibration task

        # mock the signal and language models
        signal_model = mock()
        language_model = mock()
        load_model_response = [signal_model]

        # mock the behavior of execute task
        when(main).init_language_model(self.parameters).thenReturn(language_model)
        when(main).load_signal_models(directory='').thenReturn(load_model_response)
        when(main).start_task(
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_models=[signal_model],
            fake=self.fake,
        )

        execute_task(self.task, self.parameters, self.save_folder, self.alert, self.fake)

        verify(main, times=1).start_task(
            self.task,
            self.parameters,
            self.save_folder,
            language_model=language_model,
            signal_models=[signal_model],
            fake=self.fake,
        )
        verify(main, times=1).load_signal_models(directory='')
        verify(main, times=1).init_language_model(self.parameters)

    def test_execute_with_alert_enabled(self):
        expected_alert_path = f"{STATIC_AUDIO_PATH}/{self.parameters['alert_sound_file']}"
        when(main).start_task(
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )
        when(main).play_sound(expected_alert_path)

        execute_task(self.task, self.parameters, self.save_folder, True, self.fake)

        verify(main, times=1).start_task(
            self.task,
            self.parameters,
            self.save_folder,
            language_model=None,
            signal_models=[],
            fake=self.fake,
        )
        verify(main, times=1).play_sound(expected_alert_path)


if __name__ == '__main__':
    unittest.main()
