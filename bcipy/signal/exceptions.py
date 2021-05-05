
class SignalException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
