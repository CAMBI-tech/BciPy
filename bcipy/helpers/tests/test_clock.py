"""Tests for clock-related functionality"""
import time
import unittest

from pylsl import local_clock

from bcipy.helpers.clock import Clock


class CustomTime:
    """Callable object that increments a counter on each call."""

    def __init__(self, start: int = 0):
        self.val = start - 1

    def __call__(self):
        self.val = self.val + 1
        return self.val


class TestClock(unittest.TestCase):
    """Tests for monotonic clock."""

    def test_monotonic_clock(self):
        """Test default clock behavior"""
        clock = Clock()
        time1 = clock.getTime()
        self.assertTrue(time1 > 0)

        clock.reset()
        time2 = clock.getTime()
        self.assertTrue(time2 > time1, "Reset should have no effect")

        # Delta value is machine dependent. It is likely closer than this.
        self.assertAlmostEqual(
            clock.getTime(),
            local_clock(),
            delta=4,
            msg="Default implementation uses LSL local clock.")

    def test_start_from_zero(self):
        """Test that clock can be started from zero."""
        clock = Clock(start_at_zero=True)
        self.assertEqual(0, round(clock.getTime()))
        time.sleep(0.6)

        self.assertEqual(1, round(clock.getTime()))
        clock.reset()
        self.assertEqual(0, round(clock.getTime()))

    def test_clock_customization(self):
        """Test that clock can be customized"""

        clock = Clock(get_time=CustomTime(start=0))
        self.assertEqual(1, clock.getTime())
        self.assertEqual(2, clock.getTime())
