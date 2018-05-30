from bcipy.feedback.feedback import Feedback
from psychopy import core
import sounddevice as sd


class AuditoryFeedback(Feedback):
    """Auditory Feedback."""
    def __init__(self, parameters, clock):

        # Register Feedback Type
        self.feedback_type = 'Auditory Feedback'

        # Parameters Dictionary
        self.parameters = parameters

        # Clock
        self.clock = clock

    def administer(self, sound, fs, assertion=None):
        timing = []

        if assertion:
            pass

        time = ['auditory_feedback', self.clock.getTime()]
        sd.play(sound, fs, blocking=True)
        core.wait(1)
        timing.append(time)

        return timing
