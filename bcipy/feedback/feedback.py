REGISTERED_FEEDBACK_TYPES = ['sound', 'visual']

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class Feedback:
    """Feedback."""
    def __init__(self, feedback_type):
        super(Feedback, self).__init__()
        self.feedback_type = feedback_type
        self.logger = logging

    def configure(self):
        raise NotImplementedError()

    def administer(self):
        raise NotImplementedError()

    def _type(self):
        return self.feedback_type

    def _available_modes(self):
        return REGISTERED_FEEDBACK_TYPES
