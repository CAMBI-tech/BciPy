import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class Task(object):
    """Task."""

    def __init__(self):
        super(Task, self).__init__()
        self.logger = logging

    def configure(self):
        pass

    def execute(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()
