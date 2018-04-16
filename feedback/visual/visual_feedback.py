from feedback.feedback import Feedback


class VisualFeedback(Feedback):
	"""Visual Feedback."""
	def __init__(self, display, parameters, clock):

		# Register Feedback Type
		self.feedback_type = 'Visual Feedback'

		# Display Window
		self.display = display

		# Parameters Dictionary
		self.parameters = parameters

		# Clock
		self.clock = clock

	def administer(self, assertion=None):
		timing = []

		if assertion:
			pass
		time = ['visual_feedback', self.clock.getTime()]
		timing.append(time)

		return timing

if __name__ == "__main__":
	import argparse
	from helpers.load import load_json_parameters
	from display.display_main import init_display_window
	from psychopy import core

	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--parameters', default='parameters/parameters.json',
	                help='Parameter location. Must be in parameters directory. \
	                	  Pass as parameters/parameters.json')

	args = parser.parse_args()

	# Load a parameters file
	parameters = load_json_parameters(args.parameters)
	display = init_display_window(parameters)
	clock = core.Clock()
	# Start Visual Feedback
	visual_feedback = VisualFeedback(display=display, parameters=parameters, clock=clock)
	timing = visual_feedback.administer()
	print timing
	print visual_feedback._type()
