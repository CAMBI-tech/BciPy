
class ConnectionErr(Exception):
    def __init__(self, host, port):
        self.dErrhost = host
        self.dErrport = port
        Exception.__init__(
            self, "Connection was not established with host {0} and port {1}".format(
                self.dErrhost, self.dErrport))


class StatusCodeError(Exception):
    def __init__(self, status_code):
        self.dErrArgs = status_code
        Exception.__init__(
            self,
            "Connection was not established with and has status code {0}".format(
                self.dErrArgs))


class DockerDownError(Exception):
    def __init__(self):
        Exception.__init__(self, "Check that Docker is up and running")
