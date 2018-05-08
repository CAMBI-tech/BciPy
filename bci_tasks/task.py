
class Task(object):
    """Task."""
    def __init__(self):
        super(Task, self).__init__()

    def configure(self):
        pass

    def execute(self):
        raise NotImplementedError()

    def name(self):
        raise NotImplementedError()
