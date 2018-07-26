from bcipy.feedback.feedback import Feedback
from psychopy import visual, core
from bcipy.helpers.stimuli_generation import resize_image


class VisualFeedback(Feedback):
    """Visual Feedback."""

    def __init__(self, display, parameters, clock):

        # Register Feedback Type
        self.feedback_type = 'Visual Feedback'

        # Display Window
        self.display = display

        # Parameters Dictionary
        self.parameters = parameters
        self.font_stim = self.parameters['feedback_font']
        self.height_stim = self.parameters['feedback_stim_height']

        pos_x = self.parameters['feedback_pos_x']
        pos_y = self.parameters['feedback_pos_y']

        self.pos_stim = (pos_x, pos_y)

        self.feedback_length = self.parameters['feedback_flash_time']

        # Clock
        self.clock = clock

        self.message_color = self.parameters['feedback_message_color']
        
        self.rect = visual.Rect(win=display, width=self.height_stim, height=self.height_stim, lineColor=self.message_color, pos=(self.pos_stim), lineWidth=10, ori=0.0)
        self.rect.opacity = 0

    def administer(self, stimulus, message=None, compare_assertion=None):
        """Administer.

        Adminster visual feedback. Timing information from parameters,
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
            self.rect.draw()
            stim.draw()

            self.display.flip()
            time = ['assertion_visual_feedback', self.clock.getTime()]
        else:
            stim = self._construct_stimulus(stimulus, self.pos_stim)

            stim.draw()
            self.rect.draw()
            self.display.flip()

            time = ['visual_feedback', self.clock.getTime()]
           

        core.wait(self.feedback_length)
        timing.append(time)

        return timing

    def _construct_stimulus(self, stimulus, pos):
        if '.png' in stimulus:
            image_stim = visual.ImageStim(win=self.display,
                                    image=stimulus,
                                    mask=None,
                                    pos=pos,
                                    ori=0.0)
            image_stim.size = resize_image(stimulus, self.display.size, self.height_stim)
            self.rect.width = image_stim.size[0]
            self.rect.height = image_stim.size[1]
            self.rect.opacity = 100
            return image_stim
        else:
            return visual.TextStim(win=self.display, font=self.font_stim,
                                   text=stimulus,
                                   height=self.height_stim,
                                   pos=pos)

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
    from helpers.load import load_json_parameters
    from display.display_main import init_display_window

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters',
                        default='bcipy/parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. \
                          Pass as parameters/parameters.json')

    args = parser.parse_args()

    # Load a parameters file
    parameters = load_json_parameters(args.parameters)
    display = init_display_window(parameters)
    clock = core.Clock()
    # Start Visual Feedback
    visual_feedback = VisualFeedback(
        display=display, parameters=parameters, clock=clock)
    stimulus = 'Visual'
    timing = visual_feedback.administer(stimulus)
    print(timing)
    print(visual_feedback._type())
