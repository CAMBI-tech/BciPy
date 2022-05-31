
class SignalException(Exception):
    """
    Signal Exception.

    Thrown when signal model is used improperly.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
