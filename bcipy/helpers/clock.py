"""Functionality related to clocks to keep track of time in experiments."""

from typing import Callable

from pylsl import local_clock


# Adapted from psychopy.
# pylint: disable=invalid-name
class Clock():
    """Monotonic clock that provides high resolution timestamps.

    Parameters
    ----------
        start_at_zero : If `True` the value returned by `getTime` is
            relative to when the clock was started or reset. Default is `False`
        get_time : optional function called to get the current time. The
            default value is the LabStreamingLayer (LSL) local_clock. This is
            the clock used by the acquisition module when sampling data and
            allows us to make time comparisons without conversion.
    """

    def __init__(self,
                 start_at_zero: bool = False,
                 get_time: Callable[[], float] = local_clock):
        """Initialize the clock."""
        self.start_at_zero = start_at_zero
        self._get_time = get_time
        self._last_reset_time = self._get_time()

    def getTime(self) -> float:
        """Returns the current time on this clock in seconds (sub-ms precision).

        This value is whatever the underlying clock uses as its base time but
        is system dependent. e.g. can be time since reboot, time since Unix
        Epoch etc.

        If the start_at_zero property is true, the returned value will be the
        time since the clock was created or reset.
        """
        if self.start_at_zero:
            return self._get_time() - self._last_reset_time
        return self._get_time()

    def reset(self) -> None:
        """Resets the clock."""
        self._last_reset_time = self._get_time()
