class InvalidClockError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

class UnsupportedContentType(Exception):
    """Error that occurs when attempting to collect data from a device with a
    content type that is not yet supported by BciPy."""