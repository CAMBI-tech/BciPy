"""Handles button press interactions.

This module provides classes for handling button press events in the BciPy system.
It includes abstract base classes and concrete implementations for different types
of button press handling strategies.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from psychopy import event
from psychopy.core import CountdownTimer

from bcipy.helpers.clock import Clock
from bcipy.helpers.task import get_key_press


class ButtonPressHandler(ABC):
    """Handles button press events.

    This is an abstract base class that defines the interface for handling button press
    events. It provides functionality for waiting for and processing button presses
    within a specified time window.

    Attributes:
        max_wait (float): Maximum number of seconds to wait for a key press.
        key_input (str): Key that we are listening for.
        clock (Clock): Clock used to associate the key event with a timestamp.
        response (Optional[List]): List containing the latest response information.
        _timer (Optional[CountdownTimer]): Timer for tracking wait period.
        make_timer (Type[CountdownTimer]): Factory for creating timer instances.
    """

    def __init__(
        self,
        max_wait: float,
        key_input: str,
        clock: Optional[Clock] = None,
        timer: Type[CountdownTimer] = CountdownTimer
    ) -> None:
        """Initialize the ButtonPressHandler.

        Args:
            max_wait (float): Maximum number of seconds to wait for a key press.
            key_input (str): Key that we are listening for.
            clock (Optional[Clock]): Clock used to associate the key event with a timestamp.
                If None, a new Clock instance will be created.
            timer (Type[CountdownTimer]): Factory for creating timer instances.
                Defaults to CountdownTimer.
        """
        self.max_wait = max_wait
        self.key_input = key_input
        self.clock = clock or Clock()
        self.response: Optional[List[Any]] = None

        self._timer: Optional[CountdownTimer] = None
        self.make_timer = timer

    @property
    def response_label(self) -> Optional[str]:
        """Get the label for the latest button press.

        Returns:
            Optional[str]: The label of the latest button press, or None if no response.
        """
        return self.response[0] if self.response else None

    @property
    def response_timestamp(self) -> Optional[float]:
        """Get the timestamp for the latest button response.

        Returns:
            Optional[float]: The timestamp of the latest button press, or None if no response.
        """
        return self.response[1] if self.response else None

    def _reset(self) -> None:
        """Reset any existing events and timers.

        This method clears any existing keyboard events, resets the timer,
        and clears the current response.
        """
        self._timer = self.make_timer(self.max_wait)
        self.response = None
        event.clearEvents(eventType='keyboard')
        self._timer.reset()

    def await_response(self) -> None:
        """Wait for a button response for a maximum number of seconds.

        Wait period could end early if the class determines that some other
        criteria have been met (such as an acceptable response).
        """
        self._reset()
        while self._should_keep_waiting() and self._within_wait_period():
            self._check_key_press()

    def has_response(self) -> bool:
        """Check whether a response has been provided.

        Returns:
            bool: True if a response has been provided, False otherwise.
        """
        return self.response is not None

    def _check_key_press(self) -> None:
        """Check for any key press events and set the latest as the response.

        This method updates the response attribute with the latest key press
        information if a valid key press is detected.
        """
        self.response = get_key_press(
            key_list=[self.key_input],
            clock=self.clock,
        )

    def _within_wait_period(self) -> bool:
        """Check that we are within the allotted time for a response.

        Returns:
            bool: True if we are still within the wait period, False otherwise.
        """
        return (self._timer is not None) and (self._timer.getTime() > 0)

    def _should_keep_waiting(self) -> bool:
        """Check that we should keep waiting for responses.

        Returns:
            bool: True if we should continue waiting, False otherwise.
        """
        return not self.has_response()

    @abstractmethod
    def accept_result(self) -> bool:
        """Determine if the result of a button press should be affirmative.

        Returns:
            bool: True if the result should be considered affirmative, False otherwise.
        """
        pass


class AcceptButtonPressHandler(ButtonPressHandler):
    """ButtonPressHandler where a matching button press indicates an affirmative result.

    This handler considers a button press as an affirmative response.
    """

    def accept_result(self) -> bool:
        """Determine if the result of a button press should be affirmative.

        Returns:
            bool: True if a response has been provided, False otherwise.
        """
        return self.has_response()


class RejectButtonPressHandler(ButtonPressHandler):
    """ButtonPressHandler where a matching button press indicates a rejection.

    This handler considers a button press as a rejection response.
    """

    def accept_result(self) -> bool:
        """Determine if the result of a button press should be affirmative.

        Returns:
            bool: True if no response has been provided, False otherwise.
        """
        return not self.has_response()


class PreviewOnlyButtonPressHandler(ButtonPressHandler):
    """ButtonPressHandler that waits for the entire span of the configured max_wait.

    This handler always waits for the full duration regardless of button presses.
    """

    def _should_keep_waiting(self) -> bool:
        """Check that we should keep waiting for responses.

        Returns:
            bool: Always returns True to ensure full wait duration.
        """
        return True

    def accept_result(self) -> bool:
        """Determine if the result of a button press should be affirmative.

        Returns:
            bool: Always returns True.
        """
        return True
