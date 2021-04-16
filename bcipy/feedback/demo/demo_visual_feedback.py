from bcipy.feedback.visual.visual_feedback import VisualFeedback
from psychopy import core
from bcipy.helpers.load import load_json_parameters
from bcipy.display.display_main import init_display_window


# Load a parameters file
parameters = load_json_parameters(
    'bcipy/parameters/parameters.json', value_cast=True)
display = init_display_window(parameters)
clock = core.Clock()
# Start Visual Feedback
visual_feedback = VisualFeedback(
    display=display, parameters=parameters, clock=clock)
stimulus = 'A'
message = 'Selected:'
visual_feedback.message_color = 'white'
timing = visual_feedback.administer(
    stimulus, message=message)
print(timing)
print(visual_feedback._type())
display.close()
