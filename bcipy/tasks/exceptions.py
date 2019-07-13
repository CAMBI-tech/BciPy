
class InsufficientDataException(Exception):
    """Insufficient Data Exception.

    Thrown when data requirments to execute task are violated.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class TaskRegistryException(Exception):
    """Task Registry Exception.

    Thrown when task type unregistered.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
