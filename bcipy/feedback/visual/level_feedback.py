from typing import Tuple

from psychopy import core, visual

from bcipy.feedback.visual.visual_feedback import VisualFeedback


class LevelFeedback(VisualFeedback):
    """Level Feedback.

    A progress bar like feedback to indicate current abilities in a BCI task. This could
        be attentiveness, muscle or eye activity, etc.

    It does not return stimuli timing or allow for parameterized configuration of levels / color gradient
    at this time.
    """

    def __init__(self, display, parameters, clock):
        super().__init__(display, parameters, clock)

        # extract needed parameters
        self.clock = clock
        self.display = display
        self.parameters = parameters
        self.presentation_time = parameters['feedback_duration']

        # set stylistic elements
        self.default_message_color = parameters['feedback_color']
        self.feedback_indicator_color = parameters['feedback_color']
        self.position_x = parameters['feedback_pos_x']
        self.position_y = parameters['feedback_pos_y']
        self.padding = parameters['feedback_padding']
        self.width = parameters['feedback_stim_width']
        self.height = parameters['feedback_stim_height']
        self.line_color = parameters['feedback_line_color']
        self.line_width = parameters['feedback_line_width']
        self.font = parameters['feedback_font']
        self.target_line_width = parameters['feedback_target_line_width']

        # colors needed for each level. This defines the number of levels in
        # the bar indicator.
        self.level_colors = [
            'red',
            'darkorange',
            'yellow',
            'lightgreen',
            'green']

        # array to hold our level feedback stimuli to be presented
        self.stimuli = []

    def administer(self, position=1):
        # construct the bar indicator
        self._construct_bar_indicator()

        # indexing starts at zero but position does not ;)
        position -= 1

        # loop through the levels wanted and construct the stimuli
        for index, color in enumerate(self.level_colors):
            # this is the target indicator level and should stand out of the
            # rest
            if position == index:
                self.stimuli[index].lineColor = self.feedback_indicator_color
                self.stimuli[index].lineWidth = self.target_line_width
            # draw other stimuli elements
            self.stimuli[index].fillColor = color
            self.stimuli[index].draw()

        # flip the display and present for a time
        time = ['visual_feedback', self.clock.getTime()]
        self.display.flip()
        core.wait(self.presentation_time)

        return time

    def _construct_bar_indicator(self):
        """Construct Bar Indicator.

        Helper method for constructing the bar indicator programmatically. It
            resets the position_x and stimuli variables on first call.
        """
        self._reset_stimuli()
        width, height = self._determine_height_and_width()

        # construct the rectangular level shapes and append to stimuli array
        for level in self.level_colors:
            position = self._determine_stimuli_position()
            rect = visual.Rect(
                win=self.display,
                width=width,
                height=height,
                fillColor=self.default_message_color,
                lineColor=self.line_color,
                pos=position,
                lineWidth=self.line_width,
                ori=0.0)
            self.stimuli.append(rect)

    def _determine_stimuli_position(self) -> Tuple[int, int]:
        """Determine Stimuli Position.

        Defines the x and y position of the feedback level stimuli,
            incrementing by the self.padding variable after setting.
            If a vertical bar is desired the position_y should be incremented
            instead of position_x.
        """
        # define the positioning of stimuli and increment the x position
        response = self.position_x, self.position_y
        self.position_x += self.padding
        return response

    def _determine_height_and_width(self) -> Tuple[float, float]:
        """Determine Height and Width.

        Defines the height and width of level stimuli. Currently, a static
            set of variables.
        """
        return self.width, self.height

    def _reset_stimuli(self) -> None:
        """Reset Stimuli.

        Used when redrawing stimuli. Sets the stimuli array and starting
            x positions to their starting values. If using vertical bars
            this should reset the y positions instead.
        """
        # reset stimuli
        self.stimuli = []
        self.position_x = self.parameters['feedback_pos_x']
