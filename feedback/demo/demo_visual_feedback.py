from feedback.visual.visual_feedback import VisualFeedback
from psychopy import core
from helpers.load import load_json_parameters
from display.display_main import init_display_window


# Load a parameters file
parameters = load_json_parameters('parameters/parameters.json')
display = init_display_window(parameters)
clock = core.Clock()
# Start Visual Feedback
visual_feedback = VisualFeedback(
    display=display, parameters=parameters, clock=clock)
stimulus = 'A'
assertion = 'B'
message = 'Incorrect:'
visual_feedback.message_color = 'red'
timing = visual_feedback.administer(
    stimulus, compare_assertion=assertion, message=message)
print(timing)
print(visual_feedback._type())
