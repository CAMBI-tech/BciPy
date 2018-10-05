import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class Task(object):
    """Task."""

    def __init__(self):
        super(Task, self).__init__()
        self.logger = logging

    @classmethod
    def label(cls):
        """Label to be displayed in GUI and command line interfaces."""
        obj = cls.__new__(cls)
        return obj.name()

    def configure(self):
        pass

    def execute(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()
