from feedback.feedback import Feedback
import soundfile as sf

class AuditoryFeedback(Feedback):
	"""Auditory Feedback."""
	def __init__(self, sound, fs, parameters, clock):

		# Register Feedback Type
		self.feedback_type = 'Auditory Feedback'

		# Sound (should have play method)
		self.sound = sound
		self.fs = fs
		self._sd = __import__('sounddevice')

		# Parameters Dictionary
		self.parameters = parameters

		# Clock
		self.clock = clock

	def administer(self, assertion=None):
		timing = []

		if assertion:
			pass

		time = ['auditory_feedback', self.clock.getTime()]
		self._sd.play(self.sound, self.fs, blocking=True)
		timing.append(time)

		return timing
