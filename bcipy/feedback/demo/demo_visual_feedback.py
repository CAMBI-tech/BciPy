from bcipy.display import init_display_window
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.clock import Clock
from bcipy.helpers.load import load_json_parameters

# Load a parameters file
parameters = load_json_parameters(
    'bcipy/parameters/parameters.json', value_cast=True)
display = init_display_window(parameters)
clock = Clock()
# Start Visual Feedback
visual_feedback = VisualFeedback(
    display=display, parameters=parameters, clock=clock)
stimulus = 'Selected: A'
visual_feedback.message_color = 'white'
timing = visual_feedback.administer(stimulus)
print(timing)
print(visual_feedback._type())
display.close()
