# mypy: disable-error-code=override
"""Feedback module.

This module provides the base feedback functionality for BciPy, including abstract
classes and utilities for creating and managing different types of feedback
mechanisms (sound, visual, etc.).
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List

from bcipy.config import SESSION_LOG_FILENAME


class FeedbackType(Enum):
    """Enumeration of feedback types supported by BciPy (Visual, Audio)."""

    VIS = 'Visual'
    AUD = 'Audio'

    @classmethod
    def list(cls) -> List[str]:
        """Return a list of all available feedback types.

        Returns:
            List[str]: List of feedback type values
        """
        return [feedback_type.value for feedback_type in cls]


class StimuliType(Enum):
    """Enumeration of stimuli types supported by BciPy (Text, Image)."""

    TEXT = 'Text'
    IMAGE = 'Image'

    @classmethod
    def list(cls) -> List[str]:
        """Return a list of all available stimuli types.

        Returns:
            List[str]: List of stimuli type values
        """
        return [stimuli_type.value for stimuli_type in cls]


class Feedback(ABC):
    """Abstract base class for feedback mechanisms.

    This class defines the interface for different types of feedback in BciPy,
    such as sound and visual feedback. It provides methods for configuration
    and administration of feedback.

    Attributes:
        feedback_type (str): Type of feedback (e.g., 'sound', 'visual').
        logger (logging.Logger): Logger instance for feedback-related events.
    """

    def __init__(self, feedback_type: FeedbackType) -> None:
        """Initialize Feedback.

        Args:
            feedback_type (str): Type of feedback to be administered.
        """
        super(Feedback, self).__init__()
        self.feedback_type = feedback_type
        self.logger = logging.getLogger(SESSION_LOG_FILENAME)

    @abstractmethod
    def administer(self, *args: Any, **kwargs: Any) -> None:
        """Administer feedback.

        This method should be implemented by subclasses to deliver the actual
        feedback to the user.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ...

    def _type(self) -> FeedbackType:
        """Get the feedback type.

        Returns:
            str: The type of feedback being administered.
        """
        return self.feedback_type

    def _available_modes(self) -> List[str]:
        """Get available feedback modes.

        Returns:
            List[str]: List of registered feedback types.
        """
        return FeedbackType.list()
