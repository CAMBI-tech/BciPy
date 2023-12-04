import unittest

from mockito import any, mock, unstub, verify, when
from mock import mock_open, patch

import psychopy

import bcipy.task.paradigm.rsvp.calibration.calibration
from bcipy.acquisition import LslAcquisitionClient
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.triggers import TriggerHandler, TriggerType
from bcipy.task.paradigm.rsvp.calibration.calibration import RSVPCalibrationTask


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
            'static_trigger_offset': 0.1,
            'stim_color': 'white',
            'stim_height': 0.6,
            'stim_jitter': 0.0,
            'stim_length': 3,
            'stim_number': 10,
            'stim_order': 'random',
            'stim_pos_x': 0.0,
            'stim_pos_y': 0.0,
            'stim_space_char': '_',
            'target_color': 'white',
            'target_positions': 'distributed',
            'task_buffer_length': 2,
            'task_color': 'white',
            'task_height': 0.1,
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
        self.daq = mock(
            {
                'device_spec': device_spec,
                'is_calibrated': True,
                'offset': lambda x: 0.0,
                'device_content_types': [ContentType.EEG],
                'clients_by_type': {
                    ContentType.EEG: self.eeg_client_mock
                }
            })
        self.temp_dir = ''
        self.model_metadata = mock({
            'device_spec': device_spec,
            'transform': mock(),
            'evidence_type': 'ERP'
        })

        self.display = mock()
        self.display.first_stim_time = 0.0
        self.mock_do_inquiry_response = [('a', 0.0), ('+', 0.1), ('b', 0.2), ('c', 0.3), ('a', 0.4)]
        when(self.display).do_inquiry(preview_calibration=False).thenReturn(self.mock_do_inquiry_response)
        when(self.display).wait_screen().thenReturn(None)

        when(bcipy.task.paradigm.rsvp.calibration.calibration).init_calibration_display_task(
            self.parameters, self.win, any(), any()).thenReturn(self.display)
        when(bcipy.task.paradigm.rsvp.calibration.calibration).trial_complete_message(any(), any()).thenReturn([])
        when(TriggerHandler).write().thenReturn()
        when(TriggerHandler).add_triggers(any()).thenReturn()

        when(psychopy.event).getKeys(keyList=any()).thenReturn(['space'])
        when(psychopy.core).wait(any()).thenReturn(None)

    def tearDown(self):
        """Override"""
        unstub()

    def test_initialize(self):
        """Test initialization"""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        verify(
            bcipy.task.paradigm.rsvp.calibration.calibration,
            times=1).init_calibration_display_task(
                self.parameters,
                self.win,
                any(),
                any())

    def test_execute(self):
        """Test task execute"""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)
        when(task).write_offset_trigger().thenReturn(None)
        when(task).write_trigger_data(any(), any()).thenReturn(None)

        task.execute()

        verify(self.display, times=self.parameters['stim_number']).do_inquiry(preview_calibration=False)
        verify(task, times=self.parameters['stim_number']).write_trigger_data(any(), any())
        verify(task, times=1).write_offset_trigger()

    def test_validate_parameters_throws_task_exception_empty_parameter(self):
        """Test validate parameters throws task exception when parameters is empty."""
        parameters = {}

        with self.assertRaises(Exception):
            with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
                RSVPCalibrationTask(
                    win=self.win,
                    daq=self.daq,
                    parameters=parameters,
                    file_save=self.temp_dir)

    def test_trigger_type_targetness(self):
        """Test trigger type targetness."""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        # non-target
        symbol = 'N'
        target = 'X'
        index = 2

        self.assertEqual(task.trigger_type(symbol, target, index), TriggerType.NONTARGET)

        # target
        symbol = 'X'
        target = 'X'
        index = 1

        self.assertEqual(task.trigger_type(symbol, target, index), TriggerType.TARGET)

    def test_trigger_type_fixation(self):
        """Test trigger type fixation."""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        # fixation
        symbol = '+'
        target = 'X'
        index = 1

        self.assertEqual(task.trigger_type(symbol, target, index), TriggerType.FIXATION)

    def test_trigger_type_prompt(self):
        """Test trigger type prompt."""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        # prompt, index = 0, otherwise it would be a target
        symbol = 'P'
        target = 'P'
        index = 0

        self.assertEqual(task.trigger_type(symbol, target, index), TriggerType.PROMPT)

    def test_trigger_type_preview(self):
        """Test trigger type preview."""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        # preview, index > 0, otherwise it would be a prompt
        symbol = 'inquiry_preview'
        target = 'P'
        index = 1

        self.assertEqual(task.trigger_type(symbol, target, index), TriggerType.PREVIEW)

    def test_write_trigger_data_first_run(self):
        """Test write trigger data when it is the first run of the task."""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        client_by_type_resp = {ContentType.EEG: self.eeg_client_mock}
        timing_mock = mock()
        timing = [('a', 0.0)]
        first_run = True
        when(self.daq).client_by_type(ContentType.EEG).thenReturn(client_by_type_resp)
        when(bcipy.task.paradigm.rsvp.calibration.calibration).offset_label('EEG').thenReturn('starting_offset')
        when(bcipy.task.paradigm.rsvp.calibration.calibration).convert_timing_triggers(
            timing, timing[0][0], any()).thenReturn(timing_mock)

        task.write_trigger_data(timing, first_run)

        verify(TriggerHandler, times=2).add_triggers(any())
        verify(self.eeg_client_mock, times=1).offset(0.0)
        verify(bcipy.task.paradigm.rsvp.calibration.calibration, times=1).offset_label(
            'EEG')
        verify(bcipy.task.paradigm.rsvp.calibration.calibration, times=1).convert_timing_triggers(
            timing, timing[0][0], any())

    def test_write_trigger_data_not_first_run(self):
        """Test write trigger data when it is not the first run of the task."""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)

        timing_mock = mock()
        timing = [('a', 0.0)]
        first_run = False
        when(bcipy.task.paradigm.rsvp.calibration.calibration).convert_timing_triggers(
            timing, timing[0][0], any()).thenReturn(timing_mock)

        task.write_trigger_data(timing, first_run)

        verify(TriggerHandler, times=1).add_triggers(any())

    def test_write_offset_trigger(self):
        """Test write offset trigger"""
        with patch('bcipy.helpers.triggers.open', mock_open(read_data=''), create=False):
            task = RSVPCalibrationTask(
                win=self.win,
                daq=self.daq,
                parameters=self.parameters,
                file_save=self.temp_dir)
        client_by_type_resp = {ContentType.EEG: self.eeg_client_mock}
        when(self.daq).client_by_type(ContentType.EEG).thenReturn(client_by_type_resp)
        when(bcipy.task.paradigm.rsvp.calibration.calibration).offset_label(
            'EEG',
            prefix='daq_sample_offset').thenReturn('daq_sample_offset')

        when(TriggerHandler).close().thenReturn()

        task.write_offset_trigger()

        verify(TriggerHandler, times=1).close()
        verify(TriggerHandler, times=1).add_triggers(any())
        verify(self.eeg_client_mock, times=1).offset(0.0)
        verify(bcipy.task.paradigm.rsvp.calibration.calibration, times=1).offset_label(
            'EEG', prefix='daq_sample_offset')


if __name__ == '__main__':
    unittest.main()
