
class UnregisteredExperimentException(Exception):
    """Unregistered Experiment.

    Thrown when experiment is not registered in the provided experiment path.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class FieldException(Exception):
    """Field Exception.

    Thrown when there is an exception relating to experimental fields.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class UnregisteredFieldException(FieldException):
    """Unregistered Field.

    Thrown when field is not registered in the provided field path.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class InvalidExperimentException(Exception):
    """Invalid Experiment Exception.

    Thrown when providing experiment data in the incorrect format.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
