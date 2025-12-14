class InvalidClockError(Exception):
    """Exception raised for invalid clock operations in acquisition."""

    def __init__(self, msg: str):
        """Initializes the InvalidClockError with a message.

        Args:
            msg (str): The error message.
        """
        super().__init__(msg)


class UnsupportedContentType(Exception):
    """Error that occurs when attempting to collect data from a device with a
    content type that is not yet supported by BciPy.
    """


class InsufficientDataException(Exception):
    """Insufficient Data Exception.

    Thrown when data requirements to execute task are violated.
    """
