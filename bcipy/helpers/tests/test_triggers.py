import os
import unittest
from unittest.mock import mock_open, patch

import psychopy
from mockito import any, mock, unstub, verify, when

from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, _calibration_trigger,
                                    apply_offsets, exclude_types,
                                    find_starting_offset, offset_device,
                                    offset_label, read, read_data,
                                    starting_offsets_by_device,
                                    trigger_decoder)


class TestCalibrationTrigger(unittest.TestCase):
    """Test Calibration Triggers.

    Unittests to assert the calibration trigger method. This is used
        during our tasks to reconcile timing between acquisition and displays.
    """

    clock = mock()
    display = mock()
    display.size = [500, 500]
    trigger_name = 'calibration_trigger'
    trigger_time = 1

    def setUp(self) -> None:
        unstub()

    def test_image_calibration_trigger(self):
        trigger_type = 'image'
        image_mock = mock()
        when(psychopy.visual).ImageStim(win=self.display,
                                        image=any(),
                                        pos=any(),
                                        mask=any(),
                                        ori=any()).thenReturn(image_mock)
        when(self.display).callOnFlip(any(), any(), any())
        when(image_mock).draw()
        when(self.display).flip()
        when(psychopy.core).wait(self.trigger_time)

        _calibration_trigger(self.clock, trigger_type, self.trigger_name,
                             self.trigger_time, self.display)

        verify(self.display, times=1).callOnFlip(any(), any(), any())
        verify(image_mock, times=1).draw()
        verify(self.display, times=1).flip()
        verify(psychopy.core, times=1).wait(self.trigger_time)

    def test_image_calibration_trigger_with_on_trigger(self):
        trigger_type = 'image'
        image_mock = mock()
        on_trigger = mock()
        when(psychopy.visual).ImageStim(win=self.display,
                                        image=any(),
                                        pos=any(),
                                        mask=any(),
                                        ori=any()).thenReturn(image_mock)
        when(self.display).callOnFlip(any(), any(), any())
        when(self.display).callOnFlip(on_trigger, self.trigger_name)
        when(image_mock).draw()
        when(self.display).flip()
        when(psychopy.core).wait(self.trigger_time)

        _calibration_trigger(self.clock, trigger_type, self.trigger_name,
                             self.trigger_time, self.display, on_trigger)

        verify(self.display, times=1).callOnFlip(any(), any(), any())
        verify(self.display, times=1).callOnFlip(on_trigger, self.trigger_name)
        verify(image_mock, times=1).draw()
        verify(self.display, times=1).flip()
        verify(psychopy.core, times=1).wait(self.trigger_time)

    def test_text_calibration_trigger(self):
        trigger_type = 'text'
        text_mock = mock()
        when(psychopy.visual).TextStim(self.display,
                                       text='').thenReturn(text_mock)
        when(self.display).callOnFlip(any(), any(), any())
        when(text_mock).draw()
        when(self.display).flip()
        when(psychopy.core).wait(self.trigger_time)

        _calibration_trigger(self.clock, trigger_type, self.trigger_name,
                             self.trigger_time, self.display)

        verify(self.display, times=1).callOnFlip(any(), any(), any())
        verify(text_mock, times=1).draw()
        verify(self.display, times=1).flip()
        verify(psychopy.core, times=1).wait(self.trigger_time)

    def test_text_calibration_trigger_with_on_trigger(self):
        trigger_type = 'text'
        text_mock = mock()
        on_trigger = mock()
        when(psychopy.visual).TextStim(self.display,
                                       text='').thenReturn(text_mock)
        when(self.display).callOnFlip(any(), any(), any())
        when(self.display).callOnFlip(on_trigger, self.trigger_name)
        when(text_mock).draw()
        when(self.display).flip()
        when(psychopy.core).wait(self.trigger_time)

        _calibration_trigger(self.clock, trigger_type, self.trigger_name,
                             self.trigger_time, self.display, on_trigger)

        verify(self.display, times=1).callOnFlip(any(), any(), any())
        verify(self.display, times=1).callOnFlip(on_trigger, self.trigger_name)
        verify(text_mock, times=1).draw()
        verify(self.display, times=1).flip()
        verify(psychopy.core, times=1).wait(self.trigger_time)

    def test_exception_invalid_calibration_trigger_type(self):
        trigger_type = 'invalid_type'
        with self.assertRaises(BciPyCoreException):
            _calibration_trigger(self.clock, trigger_type, self.trigger_name,
                                 self.trigger_time, self.display)

    def test_exception_no_display_calibration_trigger_type(self):
        trigger_type = 'image'
        with self.assertRaises(BciPyCoreException):
            _calibration_trigger(
                self.clock,
                trigger_type,
                self.trigger_name,
                self.trigger_time,
                None,
            )


class TestTrigger(unittest.TestCase):
    def setUp(self):
        self.label = 'A'
        self.type = TriggerType.NONTARGET
        self.time = 1
        self.test_trigger = Trigger(self.label,
                                    self.type,
                                    self.time)

    def test_create_trigger(self):
        self.assertTrue(self.test_trigger.label == self.label and
                        self.test_trigger.type == self.type and
                        self.test_trigger.time == self.time)

    def test_print_trigger(self):
        expected = f'Trigger: label=[{self.label}] type=[{self.type}] time=[{self.time}]'
        result = self.test_trigger.__repr__()
        self.assertEqual(expected, result)

    def test_from_list(self):
        trg = Trigger('A', TriggerType.NONTARGET, 1)
        self.assertEqual(Trigger.from_list(['A', 'nontarget', '1']), trg)

    def test_with_offset(self):
        trg = Trigger('A', TriggerType.NONTARGET, 1)
        self.assertEqual(trg.with_offset(1).time, 2)


class TestTriggerHandler(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    def setUp(self, mock_file):
        self.mock_file = mock_file
        self.path_name = '.'
        self.file_name = 'test'
        self.flush = FlushFrequency.END
        self.file = f'{self.path_name}/{self.file_name}.txt'
        # with patch('builtins.open', mock_open(read_data='data')) as _:
        self.handler = TriggerHandler(self.path_name, self.file_name, self.flush)
        self.mock_file.assert_called_once_with(self.file, 'w+', encoding=self.handler.encoding)

    def tearDown(self):
        unstub()

    def test_file_exist_exception(self):
        with open(self.file, 'w+', encoding=self.handler.encoding) as _:
            with self.assertRaises(Exception):
                TriggerHandler(self.path_name, self.file_name, FlushFrequency.END)
        os.remove(self.file)

    def test_add_triggers_returns_list_of_triggers(self):
        trigger = Trigger('A', TriggerType.NONTARGET, 1)
        inquiry_triggers = [trigger]

        response = self.handler.add_triggers(inquiry_triggers)
        self.assertEqual(response, inquiry_triggers)

    def test_write_triggers_flushes_triggers(self):
        inquiry_triggers = [Trigger('A', TriggerType.NONTARGET, 1)]
        self.handler.add_triggers(inquiry_triggers)
        self.assertNotEqual(self.handler.triggers, [])
        self.handler.write()
        self.assertEqual(self.handler.triggers, [])

    def test_add_triggers_calls_write_when_flush_every(self):
        self.handler.flush = FlushFrequency.EVERY
        when(self.handler).write().thenReturn()
        inquiry_triggers = [Trigger('A', TriggerType.NONTARGET, 1)]
        self.handler.add_triggers(inquiry_triggers)

        verify(self.handler, times=1).write()

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_load_returns_list_of_triggers(self, path_exists_mock):
        """Test static load method"""
        path_exists_mock.returnValue = True
        trg_data = '''A nontarget 1.0'''

        trg = Trigger('A', TriggerType.NONTARGET, 1.0)
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            response = self.handler.load('test_path_not_real.txt')
            self.assertEqual(response[0], trg)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_load_applies_offset(self, path_exists_mock):
        """Test that static load method applies offsets"""
        path_exists_mock.returnValue = True
        trg_data = '''starting_offset offset 0.5
                    A nontarget 1.0'''
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            response = self.handler.load('test_path_not_real.txt', offset=1)
            self.assertEqual(response[0].time, 2.5)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_load_exclusion(self, path_exists_mock):
        """Test that static load can exclude trigger types"""
        path_exists_mock.returnValue = True
        trg_data = '''A nontarget 1
                      + fixation 2'''
        fixation_trg = Trigger('+', TriggerType.FIXATION, 2)
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            response = self.handler.load('test_path_not_real.txt',
                                         exclusion=[TriggerType.NONTARGET])
            self.assertEqual(response[0], fixation_trg)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_read_data(self, path_exists_mock):
        """Test that trigger data is correctly read."""
        trg_data = '''starting_offset offset 3.47
                    J prompt 6.15
                    + fixation 8.11
                    F nontarget 8.58
                    D nontarget 8.88
                    J target 9.18
                    T nontarget 9.49
                    K nontarget 9.79
                    _ nontarget 11.30'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            triggers, offset = TriggerHandler.read_text_file('triggers.txt')
            self.assertEqual(len(triggers), 9)
            self.assertEqual(triggers[1].label, 'J')
            self.assertEqual(triggers[1].type, TriggerType.PROMPT)
            self.assertEqual(triggers[1].time, 6.15)
            self.assertEqual(offset, 3.47)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_read_data_bad_format(self, path_exists_mock):
        """Test that exception is thrown when trigger type doesn't exist."""
        trg_data = '''start_offset offset
                    + fixation 8.11
                    F hello_world 8.58
                    system_data system 6.23'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            with self.assertRaises(BciPyCoreException) as ctx:
                _trg, _ = TriggerHandler.read_text_file('triggers.txt')

            self.assertIn("line 1", ctx.exception.message)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_read_data_bad_trigger_type(self, path_exists_mock):
        """Test that exception is thrown when trigger type doesn't exist."""
        trg_data = '''start_offset offset 3.47
                    + fixation 8.11
                    F hello_world 8.58
                    system_data system 6.23'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            with self.assertRaises(BciPyCoreException) as ctx:
                _trg, _ = TriggerHandler.read_text_file('triggers.txt')

            self.assertIn("line 3", ctx.exception.message)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_read_data_bad_timestamp(self, path_exists_mock):
        """Test that exception is thrown when timestamp can't be converted."""
        trg_data = '''starting_offset offset 3.47
                    J prompt 6.15abc
                    + fixation 8.11
                    F hello_world 8.58
                    offset offset_correction 6.23'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            with self.assertRaises(BciPyCoreException) as ctx:
                _trg, _ = TriggerHandler.read_text_file('triggers.txt')

            self.assertIn("line 2", ctx.exception.message)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_read_data_unicode(self, path_exists_mock):
        """Test that trigger data is correctly read."""
        trg_data = '''starting_offset offset 3.47
                    ■ prompt 6.15
                    + fixation 8.11
                    F nontarget 8.58
                    D nontarget 8.88
                    ■ target 9.18
                    T nontarget 9.49
                    K nontarget 9.79
                    _ nontarget 11.30'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            triggers, offset = TriggerHandler.read_text_file('triggers.txt')
            self.assertEqual(len(triggers), 9)
            self.assertEqual(triggers[1].label, '■')
            self.assertEqual(triggers[1].type, TriggerType.PROMPT)
            self.assertEqual(triggers[1].time, 6.15)
            self.assertEqual(offset, 3.47)


class TestTriggerFunctions(unittest.TestCase):
    """Test module functions"""

    def test_starting_offset_label(self):
        """Test functions for generating offset labels"""
        self.assertEqual(offset_label(), 'starting_offset')
        self.assertEqual(offset_label('EEG'), 'starting_offset')
        self.assertEqual(offset_label('EYETRACKER'),
                         'starting_offset_EYETRACKER')

        self.assertEqual(offset_label(prefix='daq_offset'), 'daq_offset')
        self.assertEqual(offset_label(device_type='EEG', prefix='daq_offset'),
                         'daq_offset')
        self.assertEqual(
            offset_label(device_type='EYETRACKER', prefix='daq_offset'),
            'daq_offset_EYETRACKER')

    def test_offset_device_type(self):
        """Test functions for generating offset labels"""
        self.assertEqual(offset_device('starting_offset'), 'EEG')
        self.assertEqual(offset_device('starting_offset_EEG'), 'EEG')
        self.assertEqual(offset_device('starting_offset_EYETRACKER'),
                         'EYETRACKER')
        self.assertEqual(offset_device('starting_offset_EYE_TRACKER'),
                         'EYE_TRACKER')

        self.assertEqual(
            offset_device('daq_offset_EYE_TRACKER', prefix='daq_offset'),
            'EYE_TRACKER')

        with self.assertRaises(AssertionError):
            offset_device('offset', prefix='starting_offset')

    def test_read_data(self):
        """Test reading data from a given source"""
        trg_data = '''starting_offset offset 3.47
                    J prompt 6.15
                    + fixation 8.11
                    F nontarget 8.58
                    D nontarget 8.88
                    J target 9.18
                    T nontarget 9.49
                    K nontarget 9.79
                    _ nontarget 11.30'''
        triggers = read_data(trg_data.split('\n'))
        self.assertEqual(len(triggers), 9)
        self.assertEqual(triggers[1].label, 'J')
        self.assertEqual(triggers[1].type, TriggerType.PROMPT)
        self.assertEqual(triggers[1].time, 6.15)

    def test_read_data_exception(self):
        """Invalid triggers should throw an exception"""
        trg_data = '''starting_offset offset 3.47
                    J prompt 6.15
                    + fixation 8.11
                    foobarbaz'''
        with self.assertRaises(BciPyCoreException):
            read_data(trg_data.split('\n'))

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_read_from_file(self, path_exists_mock):
        """Test reading triggers from a file"""
        trg_data = '''starting_offset offset 3.47
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18
            T nontarget 9.49
            K nontarget 9.79
            _ nontarget 11.30'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data), create=True):
            triggers = read('triggers.txt')
            self.assertEqual(len(triggers), 9)
            self.assertEqual(triggers[1].label, 'J')
            self.assertEqual(triggers[1].type, TriggerType.PROMPT)
            self.assertEqual(triggers[1].time, 6.15)

    def test_find_starting_offset_default(self):
        """Test default behavior for find_starting_offset"""
        triggers = read_data('''J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18
            T nontarget 9.49
            K nontarget 9.79
            _ nontarget 11.30'''.split('\n'))
        offset = find_starting_offset(triggers)
        self.assertEqual(offset.time, 0.0)

    def test_find_starting_offset_default_device(self):
        """Test behavior for find_starting_offset when not given a device"""
        triggers = read_data('''starting_offset offset 3.47
            starting_offset_EYETRACKER offset 4.47
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18
            T nontarget 9.49
            K nontarget 9.79
            _ nontarget 11.30'''.split('\n'))
        offset = find_starting_offset(triggers)
        self.assertEqual(offset.label, 'starting_offset')
        self.assertEqual(offset.time, 3.47)

    def test_find_starting_offset_for_device(self):
        """Test behavior for find_starting_offset given a device"""
        triggers = read_data('''starting_offset offset 3.47
            starting_offset_EYETRACKER offset 4.47
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18
            T nontarget 9.49
            K nontarget 9.79
            _ nontarget 11.30'''.split('\n'))
        offset = find_starting_offset(triggers, device_type='EYETRACKER')
        self.assertEqual(offset.label, 'starting_offset_EYETRACKER')
        self.assertEqual(offset.time, 4.47)

    def test_starting_offsets_by_device(self):
        """Test default behavior for starting_offsets_by_device"""
        triggers = read_data('''starting_offset offset 3.47
            starting_offset_EYETRACKER offset 4.47
            J prompt 6.15
            + fixation 8.11'''.split('\n'))
        offsets = starting_offsets_by_device(triggers)
        self.assertEqual(len(offsets), 2)
        self.assertEqual(offsets['EEG'].time, 3.47)
        self.assertEqual(offsets['EYETRACKER'].time, 4.47)

    def test_starting_offsets_by_device_no_offset(self):
        """Test default behavior for starting_offsets_by_device with no offsets
        """
        triggers = read_data('''J prompt 6.15
            + fixation 8.11'''.split('\n'))
        offsets = starting_offsets_by_device(triggers)
        self.assertFalse(offsets)

    def test_starting_offsets_by_device_defaults(self):
        """Test default values"""
        triggers = read_data('''J prompt 6.15
            + fixation 8.11'''.split('\n'))
        offsets = starting_offsets_by_device(triggers, device_types=['EEG', 'EYETRACKER'])
        self.assertEqual(len(offsets), 2)
        self.assertEqual(offsets['EEG'].time, 0.0)
        self.assertEqual(offsets['EYETRACKER'].time, 0.0)

    def test_apply_offsets_without_static(self):
        """Test application of offsets without a static offset"""
        triggers = read_data('''starting_offset offset -3.47
            starting_offset_EYETRACKER offset -4.45
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18'''.split('\n'))
        offset = Trigger('starting_offset_EYETRACKER', TriggerType.OFFSET,
                         -4.45)
        corrected = apply_offsets(triggers, offset)
        self.assertEqual(len(corrected), 5,
                         "offset triggers should be excluded from the result")
        self.assertEqual(corrected[0].label, 'J')
        self.assertAlmostEqual(corrected[0].time, 1.7)
        self.assertEqual(corrected[-1].label, 'J')
        self.assertAlmostEqual(corrected[-1].time, 4.73)

    def test_apply_offsets_with_static(self):
        """Test application of offsets with a static offset"""
        triggers = read_data('''starting_offset offset -3.47
            starting_offset_EYETRACKER offset -4.45
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18'''.split('\n'))
        offset = Trigger('starting_offset_EYETRACKER', TriggerType.OFFSET,
                         -4.45)
        corrected = apply_offsets(triggers, offset, static_offset=0.1)
        self.assertEqual(len(corrected), 5,
                         "offset triggers should be excluded from the result")
        self.assertEqual(corrected[0].label, 'J')
        self.assertAlmostEqual(corrected[0].time, 1.8)
        self.assertEqual(corrected[-1].label, 'J')
        self.assertAlmostEqual(corrected[-1].time, 4.83)

    def test_exclude_types(self):
        """Test exclude types"""
        triggers = read_data('''starting_offset offset -3.47
            starting_offset_EYETRACKER offset -4.45
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18'''.split('\n'))
        self.assertListEqual(triggers, exclude_types(triggers, []))

        filtered = exclude_types(
            triggers,
            [TriggerType.OFFSET, TriggerType.PROMPT, TriggerType.FIXATION])
        self.assertTrue(
            all(trg.type in [TriggerType.NONTARGET, TriggerType.TARGET]
                for trg in filtered))

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_trigger_decoder_defaults(self, path_exists_mock):
        """Test trigger decoder"""
        trg_data = '''starting_offset offset -3.47
            starting_offset_EYETRACKER offset -4.45
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18
            T nontarget 9.49
            K nontarget 9.79
            _ nontarget 11.30'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            types, times, labels = trigger_decoder('triggers.txt')

            self.assertListEqual(types, [
                'nontarget', 'nontarget', 'target', 'nontarget', 'nontarget',
                'nontarget'
            ])
            self.assertListEqual(labels, ['F', 'D', 'J', 'T', 'K', '_'])
            self.assertAlmostEqual(times[0], 5.11)
            self.assertAlmostEqual(times[-1], 7.83)

    @patch('bcipy.helpers.triggers.os.path.exists')
    def test_trigger_decoder_with_device(self, path_exists_mock):
        """Test trigger decoder"""
        trg_data = '''starting_offset offset -3.47
            starting_offset_EYETRACKER offset -4.45
            J prompt 6.15
            + fixation 8.11
            F nontarget 8.58
            D nontarget 8.88
            J target 9.18
            T nontarget 9.49
            K nontarget 9.79
            _ nontarget 11.30'''
        path_exists_mock.returnValue = True
        with patch('builtins.open', mock_open(read_data=trg_data),
                   create=True):
            types, times, labels = trigger_decoder('triggers.txt',
                                                   device_type='EYETRACKER')

            self.assertListEqual(types, [
                'nontarget', 'nontarget', 'target', 'nontarget', 'nontarget',
                'nontarget'
            ])
            self.assertListEqual(labels, ['F', 'D', 'J', 'T', 'K', '_'])
            self.assertAlmostEqual(times[0], 4.13)
            self.assertAlmostEqual(times[-1], 6.85)


if __name__ == '__main__':
    unittest.main()
