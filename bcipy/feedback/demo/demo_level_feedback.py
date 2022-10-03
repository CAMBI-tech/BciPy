from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.display import init_display_window
from bcipy.feedback.visual.level_feedback import LevelFeedback
from bcipy.helpers.clock import Clock
from bcipy.helpers.load import load_json_parameters

# Load a parameters file
parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
parameters = {
    'feedback_font': parameters['font'],
    'feedback_color': parameters['info_color'],
    'feedback_pos_x': -0.5,
    'feedback_pos_y': 0,
    'feedback_padding': 0.27,
    'feedback_stim_height': 0.25,
    'feedback_stim_width': 0.25,
    'feedback_line_width': 1,
    'feedback_line_color': parameters['info_color'],
    'feedback_duration': parameters['feedback_duration'],
    'feedback_target_line_width': 5,  # this is very OS dependent!
    'full_screen': parameters['full_screen'],
    'window_height': parameters['window_height'],
    'window_width': parameters['window_width'],
    'background_color': parameters['background_color'],
    'stim_screen': parameters['stim_screen']
}
display = init_display_window(parameters)
clock = Clock()
# Start Visual Level Feedback
visual_feedback = LevelFeedback(
    display=display, parameters=parameters, clock=clock)
positions = [1, 2, 3, 4, 5]
timing = []
for i in positions:
    timing += visual_feedback.administer(position=i)
print(timing)
print(visual_feedback._type())
display.close()
