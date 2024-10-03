"""Handles button press interactions"""
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from psychopy import core, event

from bcipy.helpers.clock import Clock
from bcipy.helpers.task import get_key_press


class ButtonPressHandler(ABC):
    """Handles button press events."""

    def __init__(self,
                 max_wait: float,
                 key_input: str,
                 clock: Optional[Clock] = None,
                 timer: Type[Clock] = core.CountdownTimer):
        """
        Parameters
        ----------
            wait_length - maximum number of seconds to wait for a key press
            key_input - key that we are listening for.
            clock - clock used to associate the key event with a timestamp
        """
        self.max_wait = max_wait
        self.key_input = key_input
        self.clock = clock or Clock()
        self.response: Optional[List] = None

        self._timer = None
        self.make_timer = timer

    @property
    def response_label(self) -> Optional[str]:
        """Label for the latest button press"""
        return self.response[0] if self.response else None

    @property
    def response_timestamp(self) -> Optional[float]:
        """Timestamp for the latest button response"""
        return self.response[1] if self.response else None

    def _reset(self) -> None:
        """Reset any existing events and timers."""
        self._timer = self.make_timer(self.max_wait)
        self.response = None
        event.clearEvents(eventType='keyboard')
        self._timer.reset()

    def await_response(self):
        """Wait for a button response for a maximum number of seconds. Wait
        period could end early if the class determines that some other
        criteria have been met (such as an acceptable response)."""

        self._reset()
        while self._should_keep_waiting() and self._within_wait_period():
            self._check_key_press()

    def has_response(self) -> bool:
        """Whether a response has been provided"""
        return self.response is not None

    def _check_key_press(self) -> None:
        """Check for any key press events and set the latest as the response."""
        self.response = get_key_press(
            key_list=[self.key_input],
            clock=self.clock,
        )

    def _within_wait_period(self) -> bool:
        """Check that we are within the allotted time for a response."""
        return self._timer and self._timer.getTime() > 0

    def _should_keep_waiting(self) -> bool:
        """Check that we should keep waiting for responses."""
        return not self.has_response()

    @abstractmethod
    def accept_result(self) -> bool:
        """Should the result of a button press be affirmative"""


class AcceptButtonPressHandler(ButtonPressHandler):
    """ButtonPressHandler where a matching button press indicates an affirmative result."""

    def accept_result(self) -> bool:
        """Should the result of a button press be affirmative"""
        return self.has_response()


class RejectButtonPressHandler(ButtonPressHandler):
    """ButtonPressHandler where a matching button press indicates a rejection."""

    def accept_result(self) -> bool:
        """Should the result of a button press be affirmative"""
        return not self.has_response()


class PreviewOnlyButtonPressHandler(ButtonPressHandler):
    """ButtonPressHandler that waits for the entire span of the configured max_wait."""

    def _should_keep_waiting(self) -> bool:
        return True

    def accept_result(self) -> bool:
        """Should the result of a button press be affirmative"""
        return True
