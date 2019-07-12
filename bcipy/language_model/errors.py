
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


class EvidenceDataStructError(Exception):
    def __init__(self):
        Exception.__init__(
            self, """The evidence data structure is incorrect. It should be of
             [[(s,p),(s,p)]] or [[(s,p),(s,p)],[(s,p),(s,p)]]{s - a valid string,
              p - float}""")


class NBestError(Exception):
    def __init__(self, nbest):
        self.dErrArgs = nbest
        Exception.__init__(
            self,
            "Invalid nbest input of {0}. must be an integer".format(
                self.dErrArgs))


class NBestHighValue(Warning):
    def __init__(self, nbest):
        self.nbest = nbest
        Warning.__init__(
            self, "nbest of {0} is higher than 4 and may not perform optimally due to memory limitations".format(nbest))
