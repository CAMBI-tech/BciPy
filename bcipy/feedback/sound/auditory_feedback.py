import sounddevice as sd
from psychopy import core

from bcipy.feedback.feedback import Feedback


class AuditoryFeedback(Feedback):
    """Auditory Feedback."""

    def __init__(self, parameters, clock):

        # Register Feedback Type
        self.feedback_type = 'Auditory Feedback'

        super(AuditoryFeedback, self).__init__(self.feedback_type)

        # Parameters Dictionary
        self.parameters = parameters
        # this should not be changed. Needed to play sound correctly
        self.sound_buffer_time = 1
        self.feedback_timestamp_label = 'auditory_feedback'

        # Clock
        self.clock = clock

    def administer(self, sound, fs, assertion=None):
        timing = []

        if assertion:
            pass

        time = [self.feedback_timestamp_label, self.clock.getTime()]
        sd.play(sound, fs, blocking=True)
        core.wait(self.sound_buffer_time)
        timing.append(time)

        return timing
