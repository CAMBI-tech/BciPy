from feedback.feedback import Feedback
from psychopy import visual, core


class VisualFeedback(Feedback):
    """Visual Feedback."""

    def __init__(self, display, parameters, clock):

        # Register Feedback Type
        self.feedback_type = 'Visual Feedback'

        # Display Window
        self.display = display

        # Parameters Dictionary
        self.parameters = parameters
        self.font_stim = self.parameters['feedback_font']['value']
        self.height_stim = float(
            self.parameters['feedback_stim_height']['value'])

        pos_x = float(self.parameters['feedback_pos_x']['value'])
        pos_y = float(self.parameters['feedback_pos_y']['value'])

        self.pos_stim = (pos_x, pos_y)

        self.feedback_length = float(self.parameters[
            'feedback_flash_time']['value'])

        # Clock
        self.clock = clock

    def administer(self, stimulus, assertion=None):
        """Administer.

        Adminster visual feedback. Timing information from parameters,
            current feedback given by stimulus, if assertion type feedback
            wanted, it's added as an optional argument.
        """
        timing = []

        if assertion:
            stim, assert_stim, assert_message = self._construct_assertion_stimuli(
                stimulus, assertion)

            assert_message.draw()
            assert_stim.draw()
            stim.draw()
            self.display.flip()
            time = ['assertion_visual_feedback', self.clock.getTime()]
        else:
            stim = self._construct_stimulus(stimulus)
            stim.draw()
            self.display.flip()

            time = ['visual_feedback', self.clock.getTime()]

        core.wait(self.feedback_length)
        timing.append(time)

        return timing

    def _construct_stimulus(self, stimulus):
        if '.png' in stimulus:
            return visual.TextStim(self.display, font=self.font_stim,
                                   text=stimulus,
                                   height=self.height_stim,
                                   mask=None,
                                   pos=self.pos_sti,
                                   ori=0.0)
        else:
            return visual.TextStim(self.display, font=self.font_stim,
                                   text=stimulus,
                                   height=self.height_stim,
                                   pos=self.pos_stim)

    def _construct_assertion_stimuli(self, stimulus, assertion):
        assertion, assert_message = assertion
        stimulus = visual.TextStim(self.display, font=self.font_stim,
                                   text=stimulus,
                                   height=0.3,
                                   pos=(-.3, 0))
        assertion = visual.TextStim(self.display, font=self.font_stim,
                                    text=assertion,
                                    height=0.3,
                                    pos=(.3, 0))

        assert_message = visual.TextStim(self.display, font=self.font_stim,
                                         text=assert_message,
                                         height=0.3,
                                         pos=(-.5, .5), color='green')
        return stimulus, assertion, assert_message


if __name__ == "__main__":
    import argparse
    from helpers.load import load_json_parameters
    from display.display_main import init_display_window
    from psychopy import core

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters',
                        default='parameters/parameters.json',
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
    print timing
    print visual_feedback._type()
