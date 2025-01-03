import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mock import patch
from mockito import any, mock, unstub, verify, when

import bcipy.display.paradigm.rsvp.mode.copy_phrase
from bcipy.acquisition import LslAcquisitionClient
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.config import DEFAULT_ENCODING
from bcipy.exceptions import TaskConfigurationException
from bcipy.helpers.copy_phrase_wrapper import CopyPhraseWrapper
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import InquirySchedule
from bcipy.helpers.triggers import TriggerHandler
from bcipy.task.data import EvidenceType, Session
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask


class TestCopyPhrase(unittest.TestCase):
    """Tests for Copy Phrase task."""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        parameters = {
            'backspace_always_shown': True,
            'decision_threshold': 0.8,
            'down_sampling_rate': 2,
            'prestim_length': 1,
            'feedback_duration': 2.0,
            'feedback_stim_height': 0.35,
            'feedback_stim_width': 0.35,
            'filter_high': 45.0,
            'filter_low': 2.0,
            'filter_order': 2.0,
            'fixation_color': 'red',
            'info_color': 'white',
            'font': 'Arial',
            'info_height': 0.1,
            'info_text': '',
            'is_txt_stim': True,
            'lm_backspace_prob': 0.05,
            'max_inq_len': 50,
            'max_inq_per_series': 10,
            'max_minutes': 20,
            'min_inq_len': 5,
            'max_selections': 50,
            'max_incorrect': 10,
            'notch_filter_frequency': 60.0,
            'preview_inquiry_isi': 1.0,
            'preview_inquiry_key_input': 'space',
            'preview_inquiry_length': 5.0,
            'preview_inquiry_progress_method': 1,
            'preview_box_text_size': 0.1,
            'show_feedback': False,
            'show_preview_inquiry': False,
            'spelled_letters_count': 0,
            'stim_color': 'white',
            'rsvp_stim_height': 0.6,
            'stim_length': 10,
            'stim_number': 100,
            'stim_jitter': 0.0,
            'stim_order': 'random',
            'rsvp_stim_pos_x': 0.0,
            'rsvp_stim_pos_y': 0.0,
            'stim_space_char': '–',
            'summarize_session': False,
            'target_color': 'white',
            'task_buffer_length': 2,
            'task_color': 'white',
            'rsvp_task_height': 0.1,
            'rsvp_task_padding': 0.1,
            'task_text': 'HELLO_WORLD',
            "preview_inquiry_error_prob": 0.05,
            'info_pos_x': 0.0,
            'info_pos_y': -0.75,
            'time_fixation': 0.5,
            'time_flash': 0.25,
            'time_prompt': 1.0,
            'trial_window': (0.0, 0.5),
            'trigger_type': 'image',
        }
        self.parameters = Parameters.from_cast_values(**parameters)
        self.fake = True
        self.win = mock({'size': [500, 500], 'units': 'height'})

        device_spec = DeviceSpec(name='Testing',
                                 channels=['a', 'b', 'c'],
                                 sample_rate=300.0)
        self.eeg_client_mock = mock(
            {
                'device_spec': device_spec,
                'is_calibrated': True,
                'offset': lambda x: 0.0,
            },
            spec=LslAcquisitionClient)
        self.daq = mock(
            {
                'device_spec': device_spec,
                'is_calibrated': True,
                'offset': lambda x: 0.0,
                'device_content_types': [ContentType.EEG],
                'active_device_content_types': [ContentType.EEG],
                'clients_by_type': {
                    ContentType.EEG: self.eeg_client_mock
                }
            })
        self.servers = [mock()]
        when(self.daq).get_client(ContentType.EEG).thenReturn(self.eeg_client_mock)
        self.temp_dir = tempfile.mkdtemp()
        self.model_metadata = mock({
            'device_spec': device_spec,
            'transform': mock(),
            'evidence_type': 'ERP'
        })
        self.signal_models = [mock({'metadata': self.model_metadata})]
        self.language_model = mock()

        decision_maker = mock()
        when(decision_maker).do_series()
        self.copy_phrase_wrapper = mock({'decision_maker': decision_maker},
                                        spec=CopyPhraseWrapper)

        self.display = mock()
        self.display.first_stim_time = 0.0
        when(bcipy.task.paradigm.rsvp.copy_phrase)._init_copy_phrase_display(
            self.parameters, self.win, any(),
            any(), any()).thenReturn(self.display)

        when(bcipy.task.paradigm.rsvp.copy_phrase).CopyPhraseWrapper(
            ...).thenReturn(self.copy_phrase_wrapper)

        # mock data for initial series
        series_gen = mock_inquiry_data()
        when(self.copy_phrase_wrapper).initialize_series().thenReturn(
            next(series_gen))
        when(TriggerHandler).write().thenReturn()
        when(TriggerHandler).add_triggers(any()).thenReturn()
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).get_language_model().thenReturn(
            self.language_model)
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).get_signal_models().thenReturn(
            self.signal_models)

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)
        unstub()

    def test_initialize(self):
        """Test initialization"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        RSVPCopyPhraseTask(
            parameters=self.parameters,
            file_save=self.temp_dir,
            fake=self.fake)

    def test_validate_parameters(self):
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(
            parameters=self.parameters,
            file_save=self.temp_dir,
            fake=self.fake)

        task.validate_parameters()

    def test_validate_parameters_throws_task_exception_missing_parameter(self):
        parameters = {}
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))

        with self.assertRaises(TaskConfigurationException):
            RSVPCopyPhraseTask(
                parameters=parameters,
                file_save=self.temp_dir,

                fake=self.fake)

    def test_validate_parameters_throws_task_exception_excess_prestim_length(self):
        self.parameters['prestim_length'] = 1000
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))

        with self.assertRaises(TaskConfigurationException):
            RSVPCopyPhraseTask(
                parameters=self.parameters,
                file_save=self.temp_dir,

                fake=self.fake)

    def test_validate_parameters_throws_task_exception_excess_trial_window(self):
        self.parameters['trial_window'] = "0.0:1000.0"
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))

        with self.assertRaises(TaskConfigurationException):
            RSVPCopyPhraseTask(
                parameters=self.parameters,
                file_save=self.temp_dir,

                fake=self.fake)

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    def test_execute_without_inquiry(self, message_mock,
                                     user_input_mock):
        """User should be able to exit the task without viewing any inquiries"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        user_input_mock.return_value = False

        result = task.execute()

        # Assertions
        verify(self.copy_phrase_wrapper, times=1).initialize_series()
        verify(self.display, times=0).preview_inquiry()
        verify(self.display, times=0).do_inquiry()
        self.assertEqual(self.temp_dir, result.save_path)

        self.assertTrue(
            Path(task.session_save_location).is_file(),
            'Session data should be written')
        with open(Path(task.session_save_location), 'r', encoding=DEFAULT_ENCODING) as json_file:
            session = Session.from_dict(json.load(json_file))
            self.assertEqual(0, session.total_number_series)

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_device_data_for_decision')
    def test_execute_fake_data_single_inquiry(self, process_data_mock, message_mock,
                                              user_input_mock):
        """Test that fake data does not use the decision maker"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        # Execute a single inquiry then `escape` to stop
        user_input_mock.side_effect = [True, False]

        timings_gen = mock_inquiry_timings()
        when(self.display).do_inquiry().thenReturn(next(timings_gen))

        # mock data for single inquiry
        process_data_mock.return_value = mock_process_data()

        result = task.execute()

        # Assertions
        verify(self.copy_phrase_wrapper, times=2).initialize_series()
        verify(self.display, times=1).do_inquiry()
        self.assertEqual(self.temp_dir, result.save_path)

        self.assertTrue(
            Path(task.session_save_location).is_file(),
            'Session data should be written')
        with open(Path(task.session_save_location), 'r', encoding=DEFAULT_ENCODING) as json_file:
            session = Session.from_dict(json.load(json_file))
            self.assertEqual(1, session.total_number_series)

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_device_data_for_decision')
    def test_max_inq_len(self, process_data_mock, message_mock,
                         user_input_mock):
        """Test stoppage criteria for the max inquiry length"""
        self.parameters['max_inq_len'] = 2
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        # Don't provide any `escape` input from the user
        user_input_mock.return_value = True

        timings_gen = mock_inquiry_timings()
        when(self.display).do_inquiry().thenReturn(next(timings_gen))

        # mock data for single inquiry
        process_data_mock.return_value = mock_process_data()

        result = task.execute()

        # Assertions
        verify(self.display, times=2).do_inquiry()
        self.assertEqual(self.temp_dir, result.save_path)

        self.assertTrue(
            Path(task.session_save_location).is_file(),
            'Session data should be written')
        with open(Path(task.session_save_location), 'r', encoding=DEFAULT_ENCODING) as json_file:
            session = Session.from_dict(json.load(json_file))
            self.assertEqual(2, session.total_number_series,
                             "In fake data a decision is made every time.")
            self.assertEqual(1, len(session.series[0]))

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_device_data_for_decision')
    def test_spelling_complete(self, process_data_mock,
                               message_mock, user_input_mock):
        """Test that the task stops when the copy_phrase has been correctly spelled."""
        self.parameters['task_text'] = 'Hello'
        self.parameters['spelled_letters_count'] = 4
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        # Don't provide any `escape` input from the user
        user_input_mock.return_value = True

        timings_gen = mock_inquiry_timings()
        when(self.display).do_inquiry().thenReturn(next(timings_gen))

        # mock data for single inquiry
        process_data_mock.return_value = mock_process_data()

        result = task.execute()

        # Assertions
        verify(self.display, times=1).do_inquiry()
        self.assertEqual(self.temp_dir, result.save_path)

        self.assertTrue(
            Path(task.session_save_location).is_file(),
            'Session data should be written')
        with open(Path(task.session_save_location), 'r', encoding=DEFAULT_ENCODING) as json_file:
            session = Session.from_dict(json.load(json_file))
            self.assertEqual(1, session.total_number_series)
            self.assertEqual(1, len(session.last_series()))
            self.assertEqual('Hello',
                             session.last_inquiry().next_display_state)

    def test_spelled_letters(self):
        """Spelled letters should reset if count is larger than copy phrase."""
        self.parameters['task_text'] = 'Hi'
        self.parameters['spelled_letters_count'] = 3
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        self.assertEqual(task.starting_spelled_letters(), 0)

    def test_stims_for_eeg(self):
        """The correct stims should be sent to get_device_data_for_decision"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)
        timings1 = [['calibration_trigger', 2.0539278959913645],
                    ['+', 3.7769652379938634], ['Y', 4.247819707990857],
                    ['S', 4.46274590199755], ['W', 4.679621118993964],
                    ['T', 4.896305427988409], ['Z', 5.113133526989259],
                    ['U', 5.330021020999993], ['_', 5.5466853869875195],
                    ['V', 5.763273582997499], ['<', 5.980079917993862],
                    ['X', 6.196792346992879]]
        timings2 = [['+', 10.012395212994306], ['S', 10.480566113998066],
                    ['W', 10.698780701000942], ['_', 10.914151947989012],
                    ['T', 11.131801953000831], ['X', 11.348126853990834],
                    ['V', 11.564720113994554], ['<', 11.78139575899695],
                    ['U', 11.998181054994348], ['Z', 12.216052213989315],
                    ['Y', 12.431886781996582]]
        timings3 = [['inquiry_preview', 4.17302582100092],
                    ['bcipy_key_press_space', 5.240045738988556],
                    ['calibration_trigger', 6.246425178993377],
                    ['+', 7.9547649689920945], ['U', 8.423195326002315],
                    ['Y', 8.63952172199788], ['W', 8.85597900499124],
                    ['S', 9.072196301989607], ['V', 9.288272819001577],
                    ['_', 9.504346693996922], ['Z', 9.720611868993728],
                    ['<', 9.936859529989306], ['T', 10.181245272993692],
                    ['X', 10.414546125000925]]
        timings4 = [['inquiry_preview', 14.260656393991667],
                    ['+', 20.29594270499365], ['W', 20.76407659400138],
                    ['Z', 20.98034024599474], ['X', 21.196644898998784],
                    ['U', 21.413044401997468], ['T', 21.62935446499614],
                    ['<', 21.845442681995337], ['S', 22.06172111700289],
                    ['V', 22.277897565989406], ['Y', 22.494165399999474],
                    ['_', 22.710345731989946]]

        self.assertEqual(task.stims_for_decision(timings1), timings1[1:],
                         "calibration trigger should be omitted")
        self.assertEqual(task.stims_for_decision(timings2), timings2)
        self.assertEqual(task.stims_for_decision(timings3), timings3[3:])
        self.assertEqual(task.stims_for_decision(timings4), timings4[1:])

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_device_data_for_decision')
    def test_next_letter(self, process_data_mock, message_mock,
                         user_input_mock):
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        """Test that the task stops when the copy_phrase has been correctly spelled."""
        self.parameters['task_text'] = 'Hello'
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)
        task.spelled_text = 'H'
        self.assertEqual(task.next_target(), 'e')

        task.spelled_text = 'He'
        self.assertEqual(task.next_target(), 'l')

        task.spelled_text = 'HE'
        self.assertEqual(task.next_target(), '<', "Should be case sensitive")

        task.spelled_text = 'A'
        self.assertEqual(task.next_target(), '<')

        task.spelled_text = 'Heo'
        self.assertEqual(task.next_target(), '<')

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_device_data_for_decision')
    def test_execute_fake_data_with_preview(self, process_data_mock, message_mock,
                                            user_input_mock):
        """Test that preview is displayed"""
        self.parameters['show_preview_inquiry'] = True
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        # Execute a single inquiry then `escape` to stop
        user_input_mock.side_effect = [True, False]
        when(self.copy_phrase_wrapper).add_evidence(any, any).thenReturn([])

        timings_gen = mock_inquiry_timings()
        when(self.display).do_inquiry().thenReturn(next(timings_gen))
        when(self.display).preview_inquiry().thenReturn(([], True))

        # mock data for single inquiry
        process_data_mock.return_value = mock_process_data()

        result = task.execute()

        # Assertions
        verify(self.copy_phrase_wrapper, times=2).initialize_series()
        verify(self.display, times=1).do_inquiry()
        verify(self.copy_phrase_wrapper, times=1).add_evidence(EvidenceType.BTN, ...)
        self.assertEqual(self.temp_dir, result.save_path)

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.init_evidence_evaluator')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_user_input')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.trial_complete_message')
    @patch('bcipy.task.paradigm.rsvp.copy_phrase.get_device_data_for_decision')
    def test_execute_real_data_single_inquiry(self, process_data_mock, message_mock,
                                              user_input_mock, init_evaluator_mock):
        """Test that fake data does not use the decision maker"""
        evaluator_mock = mock()
        init_evaluator_mock.return_value = evaluator_mock

        conjugator_mock = mock({
            'latest_evidence': {
                EvidenceType.LM: [
                    0.03518519, 0.03518519, 0.03518519, 0.03518519, 0.03518519,
                    0.03518519, 0.03518519, 0.03518519, 0.03518519, 0.03518519,
                    0.03518519, 0.03518519, 0.03518519, 0.03518519, 0.03518519,
                    0.03518519, 0.03518519, 0.03518519, 0.03518519, 0.03518519,
                    0.03518519, 0.03518519, 0.03518519, 0.03518519, 0.03518519,
                    0.03518519, 0.05, 0.03518519
                ],
                EvidenceType.ERP: [
                    0.84381388, 1.18913356, 0.74758085, 1.22871603, 1.1952462,
                    1.19054715, 1.24945839, 1.17512002, 1.25628015, 1., 1., 1.,
                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                    1.20164427, 1.
                ]
            },
            'likelihood':
            np.array([
                0.02833159277115459, 0.039925922901093466,
                0.025100507130018896, 0.041254929478917374,
                0.04013115872582294, 0.0399733849464992, 0.041951367535945236,
                0.03945540913048399, 0.042180412624786785, 0.03357564204563455,
                0.03357564204563455, 0.03357564204563455, 0.03357564204563455,
                0.03357564204563455, 0.03357564204563455, 0.03357564204563455,
                0.03357564204563455, 0.03357564204563455, 0.03357564204563455,
                0.03357564204563455, 0.03357564204563455, 0.03357564204563455,
                0.03357564204563455, 0.03357564204563455, 0.03357564204563455,
                0.03357564204563455, 0.05733375793385561, 0.03357564204563455
            ])
        })
        # mock copy phrase
        copy_phrase_wrapper_mock = mock({
            'conjugator':
            conjugator_mock,
            'decision_maker':
            mock({
                'displayed_state': '',
                'last_selection': ''
            })
        })

        when(bcipy.task.paradigm.rsvp.copy_phrase).CopyPhraseWrapper(
            ...).thenReturn(copy_phrase_wrapper_mock)
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        # mock data for initial series
        when(copy_phrase_wrapper_mock).initialize_series().thenReturn(
            (False,
             InquirySchedule(
                 [['+', '<', 'G', 'A', 'E', 'H', 'D', 'F', 'I', 'B', 'C']], [[
                     0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                     0.25
                 ]], [[
                     'red', 'white', 'white', 'white', 'white', 'white',
                     'white', 'white', 'white', 'white', 'white'
                 ]])))

        when(copy_phrase_wrapper_mock).decide(...).thenReturn(
            (False,
             InquirySchedule(
                 [['+', 'E', 'I', 'F', 'G', '<', 'J', 'H', 'B', 'D', 'K']], [[
                     0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                     0.25
                 ]], [[
                     'red', 'white', 'white', 'white', 'white', 'white',
                     'white', 'white', 'white', 'white', 'white'
                 ]])))

        triggers = [['calibration_trigger', 2.5866074122022837],
                    ['+', 4.274230484152213], ['<', 4.741131300106645],
                    ['G', 4.957655837060884], ['A', 5.1744828461669385],
                    ['E', 5.391295877052471], ['H', 5.608199302107096],
                    ['D', 5.8250101080629975], ['F', 6.04189362609759],
                    ['I', 6.258658453123644], ['B', 6.475744977127761],
                    ['C', 6.692347120027989]]
        when(self.display).do_inquiry().thenReturn(triggers)

        when(copy_phrase_wrapper_mock).letter_info(any(), any()).thenReturn(
            (['<', 'G', 'A', 'E', 'H', 'D', 'F', 'I', 'B', 'C'], [
                4.741131300106645, 4.957655837060884, 5.1744828461669385,
                5.391295877052471, 5.608199302107096, 5.8250101080629975,
                6.04189362609759, 6.258658453123644, 6.475744977127761,
                6.692347120027989
            ], [
                'nontarget', 'nontarget', 'nontarget', 'nontarget',
                'nontarget', 'nontarget', 'nontarget', 'nontarget',
                'nontarget', 'nontarget'
            ]))

        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=False)

        # Execute a single inquiry then `escape` to stop
        user_input_mock.side_effect = [True, False]

        # mock data for single inquiry
        process_data_mock.return_value = mock_process_data()

        result = task.execute()

        # Assertions
        verify(copy_phrase_wrapper_mock, times=1).initialize_series()
        verify(copy_phrase_wrapper_mock, times=1).decide(...)
        verify(self.display, times=0).preview_inquiry()
        verify(self.display, times=1).do_inquiry()
        self.assertEqual(self.temp_dir, result.save_path)

        self.assertTrue(
            Path(task.session_save_location).is_file(),
            'Session data should be written')
        with open(Path(task.session_save_location), 'r', encoding=DEFAULT_ENCODING) as json_file:
            session = Session.from_dict(json.load(json_file))
            self.assertEqual(1, session.total_number_series)

    def test_btn_evidence_without_inquiry_preview_enabled(self):
        """Test button evidence without the inquiry preview functionality"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        self.parameters['show_preview_inquiry'] = False
        task = RSVPCopyPhraseTask(
            parameters=self.parameters,
            file_save=self.temp_dir,
            fake=False)
        self.assertIsNone(task.compute_button_press_evidence(True))
        self.assertIsNone(task.compute_button_press_evidence(False))

    def test_btn_evidence_without_current_inquiry(self):
        """Test button evidence without a current inquiry"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        self.parameters['show_preview_inquiry'] = True
        task = RSVPCopyPhraseTask(
            parameters=self.parameters,
            file_save=self.temp_dir,
            fake=False)
        task.current_inquiry = None
        self.assertIsNone(task.compute_button_press_evidence(True))
        self.assertIsNone(task.compute_button_press_evidence(False))

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.compute_probs_after_preview')
    def test_btn_evidence_with_inquiry_preview_enabled(self,
                                                       compute_probs_mock):
        """Test button evidence with the inquiry preview functionality"""
        probs = [
            0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.95, 0.05
        ]
        compute_probs_mock.return_value = probs
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(
            any(), any(), any()).thenReturn((self.daq, self.servers, self.win))
        self.parameters['show_preview_inquiry'] = True
        self.parameters['preview_inquiry_progress_method'] = 1
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,
                                  fake=False)

        self.assertEqual(task.compute_button_press_evidence(True),
                         (EvidenceType.BTN, probs))
        compute_probs_mock.assert_called_with(task.current_inquiry.stimuli[0],
                                              task.alp,
                                              task.button_press_error_prob,
                                              True)
        self.assertEqual(task.compute_button_press_evidence(False),
                         (EvidenceType.BTN, probs))
        compute_probs_mock.assert_called_with(task.current_inquiry.stimuli[0],
                                              task.alp,
                                              task.button_press_error_prob,
                                              False)

    def test_btn_evidence_with_preview_only(self):
        """Test button evidence with inquiry preview mode set to preview only."""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        self.parameters['show_preview_inquiry'] = True
        self.parameters['preview_inquiry_progress_method'] = 0  # ButtonPressMode.NOTHING.value

        task = RSVPCopyPhraseTask(
            parameters=self.parameters,
            file_save=self.temp_dir,
            fake=False)

        self.assertIsNone(task.compute_button_press_evidence(True))
        self.assertIsNone(task.compute_button_press_evidence(False))

    def test_setup(self):
        """Test setup"""

        when(bcipy.task.paradigm.rsvp.copy_phrase).init_acquisition(any(), any(), server=self.fake).thenReturn(
            (self.daq, self.servers))
        when(bcipy.task.paradigm.rsvp.copy_phrase).init_display_window(self.parameters).thenReturn(
            self.win)

        self.assertFalse(RSVPCopyPhraseTask.initalized)
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)

        self.assertTrue(task.initalized)
        verify(bcipy.task.paradigm.rsvp.copy_phrase, times=1).init_acquisition(
            self.parameters, self.temp_dir, server=self.fake)
        verify(bcipy.task.paradigm.rsvp.copy_phrase, times=1).init_display_window(
            self.parameters)
        self.assertEqual((self.daq, self.servers, self.win),
                         task.setup(self.parameters, self.temp_dir, self.fake))

    def test_cleanup(self):
        """Test cleanup"""
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))

        # Mock the default cleanup
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).write_offset_trigger().thenReturn(None)
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).exit_display().thenReturn(None)
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).save_session_data().thenReturn(None)
        when(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask).wait().thenReturn(None)

        # Mock the initialized cleanup
        when(self.daq).stop_acquisition().thenReturn(None)
        when(self.daq).cleanup().thenReturn(None)
        when(self.servers[0]).stop().thenReturn(None)
        when(self.win).close().thenReturn(None)
        task = RSVPCopyPhraseTask(parameters=self.parameters,
                                  file_save=self.temp_dir,

                                  fake=self.fake)
        # because the task is not initialized via setup, we need to set it to True here
        task.initalized = True

        task.cleanup()

        verify(self.daq, times=1).stop_acquisition()
        verify(self.daq, times=1).cleanup()
        verify(self.servers[0], times=1).stop()
        verify(self.win, times=1).close()
        verify(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask, times=1).setup(any(), any(), any())
        verify(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask, times=1).write_offset_trigger()
        verify(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask, times=1).exit_display()
        verify(bcipy.task.paradigm.rsvp.copy_phrase.RSVPCopyPhraseTask, times=1).wait()


def mock_inquiry_data():
    """Generator that yields data mocking the copy_phrase_wrapper initialize_series method"""
    timings = [[
        0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
    ]]
    colors = [[
        'red', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
        'white', 'white', 'white'
    ]]

    stims = [['+', 'H', 'C', 'D', '<', 'I', 'E', 'B', 'F', 'A', 'G'],
             ['+', 'I', '<', 'G', 'E', 'F', 'B', 'C', 'D', 'A', 'H'],
             ['+', 'D', 'I', 'A', 'B', 'F', 'H', 'E', '<', 'C', 'G'],
             ['+', '<', 'C', 'A', 'I', 'D', 'G', 'E', 'B', 'F', 'H'],
             ['+', 'F', 'I', 'A', 'B', 'G', 'C', 'H', 'D', 'E', '<']]

    for tup in [InquirySchedule([stim], timings, colors) for stim in stims]:
        yield (False, tup)


def mock_inquiry_timings():
    """Generator that yields data mocking the rsvp display do_inquiry() method, in response
    to stimuli that matches mock_inquiry_data."""
    timings = [[
        ['+', 4.444104908034205], ['H', 4.912202936131507],
        ['C', 5.12930660112761], ['D', 5.34570693410933],
        ['<', 5.562510692048818], ['I', 5.7791651100851595],
        ['E', 5.9961639060638845], ['B', 6.212680589174852],
        ['F', 6.429482511011884], ['A', 6.646376829128712],
        ['G', 6.863032861147076]],
        [['+', 8.669420078163967], ['H', 9.13769360119477],
         ['C', 9.35457488708198], ['D', 9.571229799184948],
         ['<', 9.788011310156435], ['I', 10.00473167304881],
         ['E', 10.221595474053174], ['B', 10.43832335807383],
         ['F', 10.655463190982118], ['A', 10.87200471619144],
         ['G', 11.088761936174706]],
        [['+', 12.864253633189946], ['H', 13.33253051317297],
         ['C', 13.549160052090883], ['D', 13.765938241034746],
         ['<', 13.98268389608711], ['I', 14.199565244140103],
         ['E', 14.416235156124458], ['B', 14.633067002054304],
         ['F', 14.850578824989498], ['A', 15.066696983063594],
         ['G', 15.284231177065521]],
        [['+', 17.058103224029765], ['H', 17.526671562111005],
         ['C', 17.743475361028686], ['D', 17.960292851086706],
         ['<', 18.176947175059468], ['I', 18.393698902102187],
         ['E', 18.610405513085425], ['B', 18.827197188977152],
         ['F', 19.04395405598916], ['A', 19.26077937404625],
         ['G', 19.477498335996643]]]
    for timing in timings:
        yield timing


def mock_process_data():
    """Generator that yields data mocking the get_device_data_for_decision helper"""
    return {ContentType.EEG: None}


def mock_triggers():
    """Mocks data from the task relative_triggers helper"""
    return [('+', 0.0), ('H', 0.4680980280973017), ('C', 0.6852016930934042),
            ('D', 0.9016020260751247), ('<', 1.1184057840146124),
            ('I', 1.335060202050954), ('E', 1.552058998029679),
            ('B', 1.7685756811406463), ('F', 1.9853776029776782),
            ('A', 2.202271921094507), ('G', 2.4189279531128705)]


if __name__ == '__main__':
    unittest.main()
