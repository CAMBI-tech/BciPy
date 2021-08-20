from enum import Enum

from psychopy import core, visual

from bcipy.feedback.feedback import Feedback
from bcipy.helpers.clock import Clock
from bcipy.helpers.stimuli import resize_image


class FeedbackType(Enum):
    TEXT = 'TEXT'
    IMAGE = 'IMAGE'


class VisualFeedback(Feedback):
    """Visual Feedback."""

    def __init__(self, display, parameters, clock):

        # Register Feedback Type
        self.feedback_type = 'Visual Feedback'

        super(VisualFeedback, self).__init__(self.feedback_type)

        # Display Window
        self.display = display

        # Parameters
        self.parameters = parameters
        self.font_stim = self.parameters['feedback_font']
        self.height_stim = self.parameters['feedback_stim_height']
        self.width_stim = self.parameters['feedback_stim_width']

        pos_x = self.parameters['feedback_pos_x']
        pos_y = self.parameters['feedback_pos_y']

        self.pos_stim = (pos_x, pos_y)

        self.feedback_length = self.parameters['feedback_flash_time']

        # Clock
        self.clock = clock

        self.message_color = self.parameters['feedback_message_color']
        self.message_pos = (-.3, .5)
        self.message_height = 0.3
        self.feedback_line_width = self.parameters['feedback_line_width']

        self.feedback_timestamp_label = 'visual_feedback'

    def administer(
            self,
            stimulus,
            fill_color='blue',
            message=None,
            stimuli_type=FeedbackType.TEXT):
        """Administer.

        Administer visual feedback. Timing information from parameters,
            current feedback given by stimulus.
        """
        timing = []

        if message:
            message = self._construct_message(message)
            message.draw()

        stim = self._construct_stimulus(
            stimulus,
            self.pos_stim,
            fill_color,
            stimuli_type)

        self._show_stimuli(stim)
        time = [self.feedback_timestamp_label, self.clock.getTime()]

        core.wait(self.feedback_length)
        timing.append(time)

        return timing

    def _show_stimuli(self, stimulus) -> None:
        stimulus.draw()
        self.display.flip()

    def _resize_image(self, stimuli, display_size, stimuli_height):
        return resize_image(
            stimulus, display_size, stimuli_height)

    def _construct_stimulus(self, stimulus, pos, fill_color, stimuli_type):
        if stimuli_type == FeedbackType.IMAGE:
            image_stim = visual.ImageStim(
                win=self.display,
                image=stimulus,
                mask=None,
                pos=pos,
                ori=0.0)
            image_stim.size = self._resize_image(
                stimulus, self.display.size, self.height_stim)
            return image_stim
        if stimuli_type == FeedbackType.TEXT:
            return visual.TextStim(
                win=self.display,
                font=self.font_stim,
                text=stimulus,
                height=self.height_stim,
                pos=pos,
                color=fill_color)

    def _construct_message(self, message):
        return visual.TextStim(
            win=self.display,
            font=self.font_stim,
            text=message,
            height=self.message_height,
            pos=self.message_pos,
            color=self.message_color)


if __name__ == "__main__":
    import argparse
    from bcipy.helpers.load import load_json_parameters
    from bcipy.display import init_display_window

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters',
                        default='bcipy/parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. \
                          Pass as bcipy/parameters/parameters.json')

    args = parser.parse_args()

    # Load a parameters file
    parameters = load_json_parameters(args.parameters, value_cast=True)
    display = init_display_window(parameters)
    clock = Clock()
    # Start Visual Feedback
    visual_feedback = VisualFeedback(
        display=display, parameters=parameters, clock=clock)
    stimulus = 'Test'
    timing = visual_feedback.administer(
        stimulus, fill_color='blue', stimuli_type=FeedbackType.TEXT)
    print(timing)
    print(visual_feedback._type())
