import os

import unittest
from unittest.mock import patch, mock_open
from mockito import any, mock, when, verify, unstub

from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.triggers import (
    _calibration_trigger,
    convert_timing_triggers,
    FlushFrequency,
    TriggerType,
    Trigger,
    TriggerHandler
)

import psychopy


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

    def test_convert_timing_triggers_returns_correct_types_given_valid_trigger_type_callable(self):
        target = 'B'
        nontarget = 'C'
        prompt = 'A'
        fixation = '+'

        # construct a trigger_type() to assign types in convert_timing_triggers()
        def trigger_type(symbol, target, index):
            if index == 0:
                return TriggerType.PROMPT
            if symbol == '+':
                return TriggerType.FIXATION
            if target == symbol:
                return TriggerType.TARGET
            return TriggerType.NONTARGET

        # This is format BciPy receives from the display which would need to be converted
        # for writing
        timing = [(prompt, 1), (fixation, 2), (target, 3), (nontarget, 4)]

        response = convert_timing_triggers(timing, target, trigger_type)
        self.assertEqual(
            response[0].type, TriggerType.PROMPT
        )
        self.assertEqual(
            response[1].type, TriggerType.FIXATION
        )
        self.assertEqual(
            response[2].type, TriggerType.TARGET
        )


class TestTriggerHandler(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="data")
    def setUp(self, mock_file):
        self.mock_file = mock_file
        self.path_name = '.'
        self.file_name = 'test'
        self.flush = FlushFrequency.END
        self.file = f'{self.path_name}/{self.file_name}.txt'
        self.handler = TriggerHandler(self.path_name, self.file_name, self.flush)
        self.mock_file.assert_called_once_with(self.file, 'w+')

    def test_file_exist_exception(self):
        with open(self.file, 'w+') as _:
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

    def test_add_triggers_calls_write_when_flush_sens_every(self):
        self.handler.flush = FlushFrequency.EVERY
        when(self.handler).write().thenReturn()
        inquiry_triggers = [Trigger('A', TriggerType.NONTARGET, 1)]
        self.handler.add_triggers(inquiry_triggers)

        verify(self.handler, times=1).write()

    def test_load_returns_list_of_triggers(self):
        txt_list = [['A', 'nontarget', '1']]
        expected = [
            Trigger(
                txt_list[0][0],
                TriggerType(txt_list[0][1]),
                float(txt_list[0][2])
            )
        ]

        when(TriggerHandler).read_text_file(any()).thenReturn((txt_list, 0.0))

        response = self.handler.load('test_path_not_real')
        self.assertEqual(response[0].label, expected[0].label)
        self.assertEqual(response[0].type, expected[0].type)
        self.assertEqual(response[0].time, expected[0].time)

    def test_load_applies_offset(self):
        txt_list = [['A', 'nontarget', '1']]
        offset = 1
        expected = [
            Trigger(
                txt_list[0][0],
                TriggerType(txt_list[0][1]),
                float(txt_list[0][2]) + offset
            )
        ]

        when(TriggerHandler).read_text_file(any()).thenReturn((txt_list, 0.0))
        response = self.handler.load('test_path_not_real', offset=offset)
        self.assertEqual(response[0].time, expected[0].time)

    def test_load_exclusion(self):
        txt_list = [['A', 'nontarget', '1'], ['A', 'fixation', '1']]
        expected = [
            Trigger(
                txt_list[1][0],
                TriggerType(txt_list[1][1]),
                float(txt_list[1][2])
            )
        ]

        exclude = TriggerType.NONTARGET
        when(TriggerHandler).read_text_file(any()).thenReturn((txt_list, 0.0))

        response = self.handler.load('test_path_not_real', exclusion=[exclude])
        self.assertEqual(response[0].label, expected[0].label)
        self.assertEqual(response[0].type, expected[0].type)
        self.assertEqual(response[0].time, expected[0].time)

    def test_load_exception_thrown_invalid_triggers_list(self):
        txt_list = [['A', 'nontarget']]
        when(TriggerHandler).read_text_file(any()).thenReturn((txt_list, 0.0))

        with self.assertRaises(BciPyCoreException):
            self.handler.load('test_path_not_real')

    def test_load_exception_thrown_invalid_trigger_type(self):
        txt_list = [['A', 'notaTriggerType', '1']]
        when(TriggerHandler).read_text_file(any()).thenReturn((txt_list, 0.0))

        with self.assertRaises(BciPyCoreException):
            self.handler.load('test_path_not_real')


if __name__ == '__main__':
    unittest.main()
