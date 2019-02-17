import logging

REGISTERED_FEEDBACK_TYPES = ['sound', 'visual']


class Feedback:
    """Feedback."""
    def __init__(self, feedback_type):
        super(Feedback, self).__init__()
        self.feedback_type = feedback_type
        self.logger = logging.getLogger(__name__)

    def configure(self):
        raise NotImplementedError()

    def administer(self):
        raise NotImplementedError()

    def _type(self):
        return self.feedback_type

    def _available_modes(self):
        return REGISTERED_FEEDBACK_TYPES
