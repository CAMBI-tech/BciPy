import logging


class Task(object):
    """Task.

    Base class for BciPy tasks.
    """

    def __init__(self):
        super(Task, self).__init__()
        self.logger = logging.getLogger(__name__)

    def configure(self):
        pass

    def execute(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()
