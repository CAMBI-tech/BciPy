from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.display import init_display_window
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.clock import Clock
from bcipy.helpers.load import load_json_parameters

# Load a parameters file
parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
parameters = {
    'feedback_font': parameters['font'],
    'feedback_color': parameters['info_color'],
    'feedback_pos_x': parameters['info_pos_x'],
    'feedback_pos_y': parameters['info_pos_y'],
    'feedback_stim_height': parameters['info_height'],
    'feedback_duration': parameters['feedback_duration'],
    'full_screen': parameters['full_screen'],
    'window_height': parameters['window_height'],
    'window_width': parameters['window_width'],
    'background_color': parameters['background_color'],
    'stim_screen': parameters['stim_screen']
}
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
