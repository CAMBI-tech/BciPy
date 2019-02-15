import logging

log = logging.getLogger(__name__)


class Task(object):
    """Task."""

    def __init__(self):
        super(Task, self).__init__()
        self.logger = log

    def configure(self):
        pass

    def execute(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()
