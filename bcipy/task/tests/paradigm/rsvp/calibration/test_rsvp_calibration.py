import unittest
from unittest.mock import Mock, patch

import psychopy
from mockito import any, mock, unstub, verify, when

import bcipy.task.calibration
import bcipy.task.paradigm.rsvp.calibration.calibration
from bcipy.acquisition import LslAcquisitionClient
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.triggers import TriggerHandler, TriggerType
from bcipy.task.paradigm.rsvp.calibration.calibration import \
    RSVPCalibrationTask


class TestRSVPCalibration(unittest.TestCase):
    """Tests for RSVP Calibration Task."""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        parameters = {
            'backspace_always_shown': True,
            'break_len': 10,
            'down_sampling_rate': 2,
            'enable_breaks': False,
            'feedback_duration': 2.0,
            'feedback_stim_height': 0.35,
            'feedback_stim_width': 0.35,
            'filter_high': 45.0,
            'filter_low': 2.0,
            'filter_order': 2.0,
            'fixation_color': 'red',
            'font': 'Arial',
            'info_color': 'white',
            'info_height': 0.1,
            'info_pos_x': 0.0,
            'info_pos_y': -0.75,
            'info_text': '',
            'is_txt_stim': True,
            'nontarget_inquiries': 0,
            'notch_filter_frequency': 60.0,
            'prestim_length': 1,
            'preview_inquiry_isi': 1.0,
            'preview_inquiry_length': 5.0,
            'show_feedback': False,
            'show_preview_inquiry': False,
            'stim_color': 'white',
            'rsvp_stim_height': 0.6,
            'stim_jitter': 0.0,
            'stim_length': 3,
            'stim_number': 10,
            'stim_order': 'random',
            'rsvp_stim_pos_x': 0.0,
            'rsvp_stim_pos_y': 0.0,
            'stim_space_char': '_',
            'target_color': 'white',
            'target_positions': 'distributed',
            'task_buffer_length': 2,
            'task_color': 'white',
            'rsvp_task_height': 0.1,
            'rsvp_task_padding': 0.1,
            'task_text': 'HELLO_WORLD',
            'time_fixation': 0.1,
            'time_flash': 0.1,
            'time_prompt': 0.1,
            'trial_window': (0.0, 0.5),
            'trials_before_break': 100,
            'trigger_type': 'image',
        }
        self.parameters = Parameters.from_cast_values(**parameters)

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
        self.daq = mock({
            'device_spec': device_spec,
            'is_calibrated': True,
            'offset': lambda x: 0.0,
            'device_content_types': [ContentType.EEG],
            'clients_by_type': {
                ContentType.EEG: self.eeg_client_mock
            }
        })
        self.temp_dir = ''
        self.fake = False
        self.servers = [mock()]
        self.model_metadata = mock({
            'device_spec': device_spec,
            'transform': mock(),
            'evidence_type': 'ERP'
        })

        self.display = mock()
        self.display.first_stim_time = 0.0
        self.mock_do_inquiry_response = [('a', 0.0), ('+', 0.1), ('b', 0.2),
                                         ('c', 0.3), ('a', 0.4)]
        when(self.display).do_inquiry(preview_calibration=False).thenReturn(
            self.mock_do_inquiry_response)
        when(self.display).wait_screen(any, any).thenReturn(None)

        when(bcipy.task.paradigm.rsvp.calibration.calibration
             ).init_calibration_display_task(self.parameters, self.win, any(),
                                             any()).thenReturn(self.display)
        when(bcipy.task.calibration).trial_complete_message(
            any(), any()).thenReturn([])
        when(bcipy.task.calibration.TriggerHandler).write().thenReturn()
        when(bcipy.task.calibration.TriggerHandler).add_triggers(
            any()).thenReturn()

        when(psychopy.event).getKeys(keyList=['space', 'escape'],
                                     modifiers=False,
                                     timeStamped=False).thenReturn(['space'])
        when(psychopy.event).getKeys(keyList=['space', 'escape']).thenReturn(
            ['space'])
        when(psychopy.core).wait(any()).thenReturn(None)

    def tearDown(self):
        """Override"""
        unstub()

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    @patch('bcipy.task.calibration.BaseCalibrationTask.cleanup')
    def test_initialize(self, save_session_mock, trigger_handler_mock, cleanup_mock):
        """Test initialization"""

        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        cleanup_mock.return_value = None

        RSVPCalibrationTask(parameters=self.parameters,
                            file_save=self.temp_dir,
                            fake=self.fake)

        verify(bcipy.task.paradigm.rsvp.calibration.calibration,
               times=1).init_calibration_display_task(self.parameters,
                                                      self.win, any(), any())

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_execute(self, save_session_mock, trigger_handler_mock):
        """Test task execute"""
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        when(task).write_offset_trigger().thenReturn(None)
        when(task).write_trigger_data(any(), any()).thenReturn(None)
        when(task).write_session_data().thenReturn(None)
        when(task).exit_display().thenReturn(None)

        task.execute()

        verify(self.display, times=self.parameters['stim_number']).do_inquiry()
        verify(task, times=self.parameters['stim_number']).write_trigger_data(
            any(), any())
        verify(task, times=1).write_offset_trigger()

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_validate_parameters_throws_task_exception_empty_parameter(
            self, save_session_mock, trigger_handler_mock):
        """Test validate parameters throws task exception when parameters is empty."""
        parameters = {}
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        with self.assertRaises(Exception):
            RSVPCalibrationTask(parameters=parameters,
                                file_save=self.temp_dir,
                                fake=self.fake)

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_trigger_type_targetness(self, save_session_mock,
                                     trigger_handler_mock):
        """Test trigger type targetness."""
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        # non-target
        symbol = 'N'
        target = 'X'
        index = 2

        self.assertEqual(task.trigger_type(symbol, target, index),
                         TriggerType.NONTARGET)

        # target
        symbol = 'X'
        target = 'X'
        index = 1

        self.assertEqual(task.trigger_type(symbol, target, index),
                         TriggerType.TARGET)

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_trigger_type_fixation(self, save_session_mock,
                                   trigger_handler_mock):
        """Test trigger type fixation."""
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        # fixation
        symbol = '+'
        target = 'X'
        index = 1

        self.assertEqual(task.trigger_type(symbol, target, index),
                         TriggerType.FIXATION)

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_trigger_type_prompt(self, save_session_mock,
                                 trigger_handler_mock):
        """Test trigger type prompt."""
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        # prompt, index = 0, otherwise it would be a target
        symbol = 'P'
        target = 'P'
        index = 0

        self.assertEqual(task.trigger_type(symbol, target, index),
                         TriggerType.PROMPT)

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_trigger_type_preview(self, save_session_mock,
                                  trigger_handler_mock):
        """Test trigger type preview."""
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        # preview, index > 0, otherwise it would be a prompt
        symbol = 'inquiry_preview'
        target = 'P'
        index = 1

        self.assertEqual(task.trigger_type(symbol, target, index),
                         TriggerType.PREVIEW)

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_write_trigger_data_first_run(self, save_session_mock,
                                          trigger_handler_mock):
        """Test write trigger data when it is the first run of the task."""
        handler_mock = Mock()
        save_session_mock.return_value = mock()
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        trigger_handler_mock.return_value = handler_mock
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        client_by_type_resp = {ContentType.EEG: self.eeg_client_mock}
        timing_mock = mock()
        timing = [('a', 0.0)]
        first_run = True
        when(self.daq).client_by_type(
            ContentType.EEG).thenReturn(client_by_type_resp)
        when(bcipy.task.calibration).offset_label('EEG').thenReturn(
            'starting_offset')
        when(bcipy.task.calibration).convert_timing_triggers(
            timing, timing[0][0], any()).thenReturn(timing_mock)

        task.write_trigger_data(timing, first_run)

        self.assertEqual(2, handler_mock.add_triggers.call_count)
        verify(self.eeg_client_mock, times=1).offset(0.0)
        verify(bcipy.task.calibration, times=1).offset_label('EEG')
        verify(bcipy.task.calibration,
               times=1).convert_timing_triggers(timing, timing[0][0], any())

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_write_trigger_data_not_first_run(self, save_session_mock,
                                              trigger_handler_mock):
        """Test write trigger data when it is not the first run of the task."""
        handler_mock = Mock()
        save_session_mock.return_value = mock()
        trigger_handler_mock.return_value = handler_mock
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        timing_mock = mock()
        timing = [('a', 0.0)]
        first_run = False
        when(bcipy.task.calibration).convert_timing_triggers(
            timing, timing[0][0], any()).thenReturn(timing_mock)

        task.write_trigger_data(timing, first_run)
        handler_mock.add_triggers.assert_called_once()

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_write_offset_trigger(self, save_session_mock,
                                  trigger_handler_mock):
        """Test write offset trigger"""
        save_session_mock.return_value = mock()
        handler_mock = Mock()
        trigger_handler_mock.return_value = handler_mock
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)
        client_by_type_resp = {ContentType.EEG: self.eeg_client_mock}
        when(self.daq).client_by_type(
            ContentType.EEG).thenReturn(client_by_type_resp)
        when(bcipy.task.calibration).offset_label(
            'EEG', prefix='daq_sample_offset').thenReturn('daq_sample_offset')

        when(TriggerHandler).close().thenReturn()

        task.write_offset_trigger()
        handler_mock.close.assert_called_once()
        handler_mock.add_triggers.assert_called_once()
        verify(self.eeg_client_mock, times=1).offset(0.0)
        verify(bcipy.task.calibration,
               times=1).offset_label('EEG', prefix='daq_sample_offset')

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_setup(self, save_session_mock, trigger_handler_mock):
        """Test setup"""
        save_session_mock.return_value = mock()
        handler_mock = Mock()
        trigger_handler_mock.return_value = handler_mock
        when(bcipy.task.calibration).init_acquisition(any(), any(), server=self.fake).thenReturn(
            (self.daq, self.servers))
        when(bcipy.task.calibration).init_display_window(self.parameters).thenReturn(
            self.win)

        self.assertFalse(RSVPCalibrationTask.initalized)
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)

        self.assertTrue(task.initalized)
        verify(bcipy.task.calibration, times=1).init_acquisition(
            self.parameters, self.temp_dir, server=self.fake)
        verify(bcipy.task.calibration, times=1).init_display_window(
            self.parameters)
        self.assertEqual((self.daq, self.servers, self.win),
                         task.setup(self.parameters, self.temp_dir, self.fake))

    @patch('bcipy.task.calibration.TriggerHandler')
    @patch('bcipy.task.calibration._save_session_related_data')
    def test_cleanup(self, save_session_mock, trigger_handler_mock):
        """Test cleanup"""
        save_session_mock.return_value = mock()
        handler_mock = Mock()
        trigger_handler_mock.return_value = handler_mock
        when(bcipy.task.calibration.BaseCalibrationTask).setup(any(), any(), any()).thenReturn(
            (self.daq, self.servers, self.win))

        # Mock the default cleanup
        when(bcipy.task.calibration.BaseCalibrationTask).write_offset_trigger().thenReturn(None)
        when(bcipy.task.calibration.BaseCalibrationTask).exit_display().thenReturn(None)
        when(bcipy.task.calibration.BaseCalibrationTask).wait().thenReturn(None)

        # Mock the initialized cleanup
        when(self.daq).stop_acquisition().thenReturn(None)
        when(self.daq).cleanup().thenReturn(None)
        when(self.servers[0]).stop().thenReturn(None)
        when(self.win).close().thenReturn(None)
        task = RSVPCalibrationTask(parameters=self.parameters,
                                   file_save=self.temp_dir,
                                   fake=self.fake)
        # because the task is not initialized via setup, we need to set it to True here
        task.initalized = True

        task.cleanup()

        verify(self.daq, times=1).stop_acquisition()
        verify(self.daq, times=1).cleanup()
        verify(self.servers[0], times=1).stop()
        verify(self.win, times=1).close()
        verify(bcipy.task.calibration.BaseCalibrationTask, times=1).setup(any(), any(), any())
        verify(bcipy.task.calibration.BaseCalibrationTask, times=1).write_offset_trigger()
        verify(bcipy.task.calibration.BaseCalibrationTask, times=1).exit_display()
        verify(bcipy.task.calibration.BaseCalibrationTask, times=1).wait()


if __name__ == '__main__':
    unittest.main()
