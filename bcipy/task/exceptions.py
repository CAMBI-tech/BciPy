"""Task-specific exceptions for the BciPy task module.

This module defines custom exceptions that can be raised during task execution,
registration, and evidence evaluation.
"""

from typing import Any, Optional


class InsufficientDataException(Exception):
    """Exception raised when task data requirements are not met.

    This exception is raised when a task does not have sufficient data to
    execute properly, such as missing calibration data or required parameters.

    Args:
        message: Description of what data was insufficient.
        errors: Optional additional error information.

    Attributes:
        message: The error message.
        errors: Additional error details, if any.
    """

    def __init__(self, message: str, errors: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors


class TaskRegistryException(Exception):
    """Exception raised when there are issues with task registration.

    This exception is raised when attempting to use an unregistered task type
    or when there are problems with the task registry.

    Args:
        message: Description of the registration issue.
        errors: Optional additional error information.

    Attributes:
        message: The error message.
        errors: Additional error details, if any.
    """

    def __init__(self, message: str, errors: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors


class MissingEvidenceEvaluator(Exception):
    """Exception raised when a required evidence evaluator is not found.

    This exception is raised when no evidence evaluator can be found that matches
    the provided data content type input and evidence_type output requirements.

    Args:
        message: Description of the missing evaluator.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DuplicateModelEvidence(Exception):
    """Exception raised when multiple models produce the same evidence type.

    This exception is raised when more than one of the provided models produces
    the same type of evidence, making it ambiguous which evidence should be used.

    Args:
        message: Description of the duplicate evidence.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
