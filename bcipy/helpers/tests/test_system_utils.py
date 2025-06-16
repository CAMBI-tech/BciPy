import socket
import unittest

import psutil
from mockito import any, mock, verify, when
from pyglet import canvas

from bcipy.helpers.utils import (ScreenInfo, get_screen_info,
                                 is_battery_powered, is_connected,
                                 is_screen_refresh_rate_low)


class TestSystemUtilsAlerts(unittest.TestCase):

    def test_is_connected_true(self):
        """Test that a computer connected to the internet returns True."""
        mock_conn = mock()
        when(socket).create_connection(address=any, timeout=any).thenReturn(mock_conn)
        self.assertTrue(is_connected())

    def test_is_connected_false(self):
        """Test that a computer not connected to the internet returns False."""
        when(socket).create_connection(address=any, timeout=any).thenRaise(OSError)
        self.assertFalse(is_connected())

    def test_is_battery_powered_true(self):
        """Test that a computer running on battery returns True."""
        mock_battery = mock()
        mock_battery.power_plugged = False
        when(psutil).sensors_battery().thenReturn(mock_battery)
        self.assertTrue(is_battery_powered())

    def test_is_battery_powered_false(self):
        """Test that a computer plugged in returns False."""
        mock_battery = mock()
        mock_battery.power_plugged = True
        when(psutil).sensors_battery().thenReturn(mock_battery)
        self.assertFalse(is_battery_powered())

    def test_is_screen_refresh_rate_low(self):
        """Test that a refresh rate of less than 120 Hz returns True."""
        display = mock()
        screen = mock()
        screen.width = 1920
        screen.height = 1080
        screen.rate = 60
        when(canvas).get_display().thenReturn(display)
        when(display).get_default_screen().thenReturn(screen)
        when(screen).get_mode().thenReturn(screen)

        self.assertTrue(is_screen_refresh_rate_low())
        verify(display, times=1).get_default_screen()
        verify(screen, times=1).get_mode()

    def test_is_screen_refresh_rate_low_false(self):
        """Test that a refresh rate of 120 Hz or greater returns False."""
        display = mock()
        screen = mock()
        screen.width = 1920
        screen.height = 1080
        screen.rate = 120
        when(canvas).get_display().thenReturn(display)
        when(display).get_default_screen().thenReturn(screen)
        when(screen).get_mode().thenReturn(screen)
        self.assertFalse(is_screen_refresh_rate_low())
        verify(display, times=1).get_default_screen()
        verify(screen, times=1).get_mode()

    def test_is_screen_refresh_ok_given_refresh(self):
        """Test that a refresh rate of 120 Hz or greater returns False."""
        self.assertFalse(is_screen_refresh_rate_low(120))

    def test_get_screen_info(self):
        """Test that the screen info is returned as expected."""
        display = mock()
        screen = mock()
        screen.width = 1920
        screen.height = 1080
        screen.rate = 120
        when(canvas).get_display().thenReturn(display)
        when(display).get_default_screen().thenReturn(screen)
        when(screen).get_mode().thenReturn(screen)
        expected = ScreenInfo(width=1920, height=1080, rate=120)
        self.assertEqual(get_screen_info(), expected)
        verify(display, times=1).get_default_screen()
        verify(screen, times=1).get_mode()


if __name__ == "__main__":
    unittest.main()
