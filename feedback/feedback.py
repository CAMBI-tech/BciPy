

REGISTERED_FEEDBACK_TYPES = ['sound', 'visual']

class Feedback(object):
	"""Feedback."""
	def __init__(self, feedback_type):
		super(Feedback, self).__init__()
		self.feedback_type = feedback_type

	def configure(self):
		pass

	def administer(self):
		pass

	def _type(self):
		return self.feedback_type

	def _available_modes(self):
		return REGISTERED_FEEDBACK_TYPES
		