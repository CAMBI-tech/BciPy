"""Custom exceptions for the BciPy package.

This module contains all custom exceptions used throughout the BciPy application.
Each exception is designed to provide specific error information for different
components of the system.
"""


class BciPyCoreException(Exception):
    """Base exception class for BciPy-specific errors.

    Args:
        message: A descriptive message explaining the error.
        errors: Optional additional error information.

    Attributes:
        message: The error message.
        errors: Additional error details, if any.
    """

    def __init__(self, message: str, errors: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors


class SignalException(BciPyCoreException):
    """Exception raised for errors in the signal processing module.

    This exception is raised when the signal module encounters errors during
    processing or analysis of signal data.

    Args:
        message: A descriptive message explaining the signal processing error.
        errors: Optional additional error information.
    """

    def __init__(self, message: str, errors: Any = None) -> None:
        super().__init__(message)
        self.errors = errors


class FieldException(BciPyCoreException):
    """Exception raised for errors related to experimental fields.

    This exception is raised when there are issues with field definitions,
    validation, or processing in experiments.

    Args:
        message: A descriptive message explaining the field-related error.
        errors: Optional additional error information.
    """
    ...


class ExperimentException(BciPyCoreException):
    """Exception raised for errors related to experiment execution.

    This exception is raised when there are issues with experiment setup,
    execution, or validation.

    Args:
        message: A descriptive message explaining the experiment-related error.
        errors: Optional additional error information.
    """
    ...


class UnregisteredExperimentException(ExperimentException):
    """Exception raised when attempting to use an unregistered experiment.

    This exception is raised when trying to access or execute an experiment
    that has not been registered in the provided experiment path.

    Args:
        message: A descriptive message explaining which experiment was not found.
        errors: Optional additional error information.
    """
    ...


class UnregisteredFieldException(FieldException):
    """Exception raised when attempting to use an unregistered field.

    This exception is raised when trying to access a field that has not been
    registered in the provided field path.

    Args:
        message: A descriptive message explaining which field was not found.
        errors: Optional additional error information.
    """
    ...


class InvalidExperimentException(ExperimentException):
    """Exception raised when experiment data is in an invalid format.

    This exception is raised when experiment configuration or data does not
    meet the required format specifications.

    Args:
        message: A descriptive message explaining the format error.
        errors: Optional additional error information.
    """
    ...


class InvalidFieldException(FieldException):
    """Exception raised when field data is in an invalid format.

    This exception is raised when field configuration or data does not
    meet the required format specifications.

    Args:
        message: A descriptive message explaining the format error.
        errors: Optional additional error information.
    """
    ...


class TaskConfigurationException(BciPyCoreException):
    """Exception raised when task configuration is invalid.

    This exception is raised when attempting to run a task with invalid
    or incompatible configuration settings.

    Args:
        message: A descriptive message explaining the configuration error.
        errors: Optional additional error information.
    """
    ...


class KenLMInstallationException(BciPyCoreException):
    """Exception raised when KenLM module is not properly installed.

    This exception is raised when attempting to use KenLM functionality
    without having the required module installed.

    Args:
        message: A descriptive message explaining the installation issue.
        errors: Optional additional error information.
    """
    ...


class InvalidSymbolSetException(BciPyCoreException):
    """Exception raised when symbol set is not properly configured.

    This exception is raised when attempting to query a language model for
    predictions without properly configuring the symbol set.

    Args:
        message: A descriptive message explaining the symbol set error.
        errors: Optional additional error information.
    """
    ...


class LanguageModelNameInUseException(BciPyCoreException):
    """Exception raised when attempting to register a duplicate language model.

    This exception is raised when trying to register a language model type
    with a name that is already in use.

    Args:
        message: A descriptive message explaining the naming conflict.
        errors: Optional additional error information.
    """
    ...
