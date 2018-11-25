from bcipy.feedback.feedback import Feedback
from psychopy import visual, core
from bcipy.helpers.stimuli_generation import resize_image
from enum import Enum


class FeedbackType(Enum):
    TEXT = 'TEXT'
    IMAGE = 'IMAGE'
    SHAPE = 'SHAPE'


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
        self.feedback_line_width = self.parameters['feedback_line_width']
        self.rectangle_message_box = self.parameters['rectangle_message_box']
        self.circular_message_box = self.parameters['circular_message_box']

        if self.rectangle_message_box:
            self.rect = visual.Rect(
                win=display,
                width=self.width_stim,
                height=self.height_stim,
                lineColor=self.message_color,
                pos=(self.pos_stim),
                lineWidth=self.feedback_line_width,
                ori=0.0)
            self.rect.opacity = 0

        if self.circular_message_box:
            self.circle = visual.Circle(
                win=display,
                radius=self.width_stim,
                lineColor=self.message_color,
                pos=(self.pos_stim),
                lineWidth=self.feedback_line_width,
                ori=0.0)

            self.circle.opacity = 0

    def administer(
            self,
            stimulus,
            pos=None,
            line_color='blue',
            fill_color='blue',
            message=None,
            compare_assertion=None,
            stimuli_type=FeedbackType.TEXT):
        """Administer.

        Administer visual feedback. Timing information from parameters,
            current feedback given by stimulus, if assertion type feedback
            wanted, it's added as an optional argument.
        """
        timing = []

        if message:
            message = self._construct_message(message)
            message.draw()

        if compare_assertion:
            (stim,
             assert_stim) = self._construct_assertion_stimuli(
                stimulus, compare_assertion)

            assert_stim.draw()
        else:
            stim = self._construct_stimulus(
                stimulus,
                self.pos_stim,
                line_color,
                fill_color,
                stimuli_type)

        self._show_stimuli(stim)
        time = ['visual_feedback', self.clock.getTime()]

        core.wait(self.feedback_length)
        timing.append(time)

        return timing

    def _show_stimuli(self, stimulus):
        if self.rectangle_message_box:
            self.rect.draw()
        if self.circular_message_box:
            self.circle.draw()
        stimulus.draw()
        self.display.flip()

    def _construct_stimulus(self, stimulus, pos, line_color, fill_color, stimuli_type):
        if stimuli_type == FeedbackType.IMAGE:
            image_stim = visual.ImageStim(
                win=self.display,
                image=stimulus,
                mask=None,
                pos=pos,
                ori=0.0)
            image_stim.size = resize_image(stimulus, self.display.size, self.height_stim)
            if self.rectangle_message_box:
                self.rect.width = image_stim.size[0]
                self.rect.height = image_stim.size[1]
                self.rect.opacity = 1
                self.rect.lineColor = self.message_color
            if self.circular_message_box:
                self.circle.radius = 1
                self.circle.opacity = 1
                self.circle.lineColor = self.message_color
            return image_stim
        if stimuli_type == FeedbackType.TEXT:
            return visual.TextStim(
                win=self.display,
                font=self.font_stim,
                text=stimulus,
                height=self.height_stim,
                pos=pos,
                color=color)
        if stimuli_type == FeedbackType.SHAPE:
            return visual.Rect(
                fillColor=fill_color, win=self.display,
                width=self.width_stim,
                height=self.height_stim,
                lineColor=line_color,
                pos=(self.pos_stim),
                lineWidth=self.feedback_line_width,
                ori=0.0)

    def _construct_assertion_stimuli(self, stimulus, assertion):
        stimulus = self._construct_stimulus(stimulus, (-.3, 0))
        assertion = self._construct_stimulus(assertion, (.3, 0))

        return stimulus, assertion

    def _construct_message(self, message):
        return visual.TextStim(win=self.display, font=self.font_stim,
                               text=message,
                               height=0.3,
                               pos=(-.3, .5), color=self.message_color)


if __name__ == "__main__":
    import argparse
    from bcipy.helpers.load import load_json_parameters
    from bcipy.display.display_main import init_display_window

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters',
                        default='bcipy/parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. \
                          Pass as bcipy/parameters/parameters.json')

    args = parser.parse_args()

    # Load a parameters file
    parameters = load_json_parameters(args.parameters, value_cast=True)
    display = init_display_window(parameters)
    clock = core.Clock()
    # Start Visual Feedback
    visual_feedback = VisualFeedback(
        display=display, parameters=parameters, clock=clock)
    stimulus = 'null'
    timing = visual_feedback.administer(
        stimulus, line_color='blue', fill_color='blue', stimuli_type=FeedbackType.SHAPE)
    print(timing)
    print(visual_feedback._type())
