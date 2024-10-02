"""Tests for button press handlers."""

import unittest
from unittest.mock import Mock, patch

from bcipy.display.components.button_press_handler import (
    AcceptButtonPressHandler, PreviewOnlyButtonPressHandler,
    RejectButtonPressHandler, get_button_handler_class)
from bcipy.display.main import ButtonPressMode


class TestButtonPressModule(unittest.TestCase):
    """Test the module functions"""

    def test_get_handler_class(self):
        """Test get handler class"""
        self.assertEqual(PreviewOnlyButtonPressHandler,
                         get_button_handler_class(ButtonPressMode.NOTHING))
        self.assertEqual(AcceptButtonPressHandler,
                         get_button_handler_class(ButtonPressMode.ACCEPT))
        self.assertEqual(RejectButtonPressHandler,
                         get_button_handler_class(ButtonPressMode.REJECT))


class TestPreviewOnlyButtonPressHandler(unittest.TestCase):
    """Test the PreviewOnlyButtonPressHandler"""

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_preview_only(self, get_key_press_mock):
        """Test preview only"""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [2.0, 1.0, 0.0]
        get_key_press_mock.return_value = None

        handler = PreviewOnlyButtonPressHandler(max_wait=2,
                                                key_input='space',
                                                timer=mock_timer_class)
        handler.await_response()

        mock_timer_class.assert_called_once()
        mock_timer.reset.assert_called_once()
        self.assertEqual(3, mock_timer.getTime.call_count)
        self.assertEqual(2, get_key_press_mock.call_count)

        self.assertTrue(handler.accept_result())
        self.assertFalse(handler.has_response())

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_preview_only_with_keypress(self, get_key_press_mock):
        """Test preview only with a keypress"""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [2.0, 1.0, 0.0]
        get_key_press_mock.side_effect = [None, ['bcipy_key_press_space', 1.5]]

        handler = PreviewOnlyButtonPressHandler(max_wait=2,
                                                key_input='space',
                                                timer=mock_timer_class)
        handler.await_response()

        mock_timer_class.assert_called_once()
        mock_timer.reset.assert_called_once()
        self.assertEqual(3, mock_timer.getTime.call_count)
        self.assertEqual(2, get_key_press_mock.call_count)

        self.assertTrue(handler.accept_result())
        self.assertTrue(handler.has_response())

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_preview_only_with_keypress_in_middle(self, get_key_press_mock):
        """Test preview only with a keypress in the middle of the wait period"""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [3.0, 2.0, 1.0, 0.0]
        get_key_press_mock.side_effect = [
            None, ['bcipy_key_press_space', 1.5], None
        ]

        handler = PreviewOnlyButtonPressHandler(max_wait=3,
                                                key_input='space',
                                                timer=mock_timer_class)
        handler.await_response()

        self.assertEqual(4, mock_timer.getTime.call_count)
        self.assertEqual(3, get_key_press_mock.call_count)

        self.assertTrue(handler.accept_result())
        self.assertFalse(handler.has_response())


class TestAcceptButtonPressHandler(unittest.TestCase):
    """Test for Press To Accept behavior"""

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_press_to_accept_with_no_presses(self, get_key_press_mock):
        """Test press to accept with no button presses."""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [2.0, 1.0, 0.0]
        get_key_press_mock.return_value = None

        handler = AcceptButtonPressHandler(max_wait=2,
                                           key_input='space',
                                           timer=mock_timer_class)
        handler.await_response()

        self.assertEqual(3, mock_timer.getTime.call_count)
        self.assertEqual(2, get_key_press_mock.call_count)

        self.assertFalse(handler.accept_result())
        self.assertFalse(handler.has_response())

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_press_to_accept_with_keypress(self, get_key_press_mock):
        """Test press to accept with a keypress"""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [3.0, 2.0, 1.0, 0.0]
        get_key_press_mock.side_effect = [None, ['bcipy_key_press_space', 1.5]]

        handler = AcceptButtonPressHandler(max_wait=2,
                                           key_input='space',
                                           timer=mock_timer_class)
        handler.await_response()

        self.assertEqual(2, mock_timer.getTime.call_count)
        self.assertEqual(2, get_key_press_mock.call_count)

        self.assertTrue(handler.accept_result())
        self.assertTrue(handler.has_response())
        self.assertEqual('bcipy_key_press_space', handler.response_label)
        self.assertEqual(1.5, handler.response_timestamp)


class TestRejectButtonPressHandler(unittest.TestCase):
    """Test for Press To Reject behavior"""

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_press_to_reject_with_no_presses(self, get_key_press_mock):
        """Test press to reject with no button presses."""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [2.0, 1.0, 0.0]
        get_key_press_mock.return_value = None

        handler = RejectButtonPressHandler(max_wait=2,
                                           key_input='space',
                                           timer=mock_timer_class)
        handler.await_response()

        self.assertEqual(3, mock_timer.getTime.call_count)
        self.assertEqual(2, get_key_press_mock.call_count)

        self.assertTrue(handler.accept_result())
        self.assertFalse(handler.has_response())

    @patch('bcipy.display.components.button_press_handler.get_key_press')
    def test_press_to_reject_with_keypress(self, get_key_press_mock):
        """Test press to accept with a keypress"""
        mock_timer_class = Mock()
        mock_timer = Mock()
        mock_timer_class.return_value = mock_timer

        # Timer counts down
        mock_timer.reset.return_value = None
        mock_timer.getTime.side_effect = [3.0, 2.0, 1.0, 0.0]
        get_key_press_mock.side_effect = [None, ['bcipy_key_press_space', 1.5]]

        handler = RejectButtonPressHandler(max_wait=2,
                                           key_input='space',
                                           timer=mock_timer_class)
        handler.await_response()

        self.assertEqual(2, mock_timer.getTime.call_count)
        self.assertEqual(2, get_key_press_mock.call_count)

        self.assertFalse(handler.accept_result())
        self.assertTrue(handler.has_response())
        self.assertEqual('bcipy_key_press_space', handler.response_label)
        self.assertEqual(1.5, handler.response_timestamp)
