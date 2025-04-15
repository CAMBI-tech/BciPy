
class BciPyCoreException(Exception):
    """BciPy Core Exception.

    Thrown when an error occurs specific to BciPy core concepts.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.message = message
        self.errors = errors


class SignalException(BciPyCoreException):
    """
    Signal Exception.

    Thrown when signal module encounters an error.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class FieldException(BciPyCoreException):
    """Field Exception.

    Thrown when there is an exception relating to experimental fields.
    """
    ...


class ExperimentException(BciPyCoreException):
    """Experiment Exception.

    Thrown when there is an exception relating to experiments.
    """
    ...


class UnregisteredExperimentException(ExperimentException):
    """Unregistered Experiment.

    Thrown when experiment is not registered in the provided experiment path.
    """

    ...


class UnregisteredFieldException(FieldException):
    """Unregistered Field.

    Thrown when field is not registered in the provided field path.
    """

    ...


class InvalidExperimentException(ExperimentException):
    """Invalid Experiment Exception.

    Thrown when providing experiment data in the incorrect format.
    """

    ...


class InvalidFieldException(FieldException):
    """Invalid Field Exception.

    Thrown when providing field data in the incorrect format.
    """

    ...


class TaskConfigurationException(BciPyCoreException):
    """Task Configuration Exception.

    Thrown when attempting to run a task with invalid configurations"""
    ...


class KenLMInstallationException(BciPyCoreException):
    """KenLM Installation Exception.

    Thrown when attempting to import kenlm without installing the module"""
    ...


class InvalidSymbolSetException(BciPyCoreException):
    """Invalid Symbol Set Exception.
    
    Thrown when querying a language model for predictions without setting the symbol set."""
    ...

class LanguageModelNameInUseException(BciPyCoreException):
    """Language Model Name In Use Exception.
    
    Thrown when attempting to register a language model type with a duplicate name."""
    ...