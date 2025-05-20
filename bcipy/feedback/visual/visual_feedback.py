# mypy: disable-error-code="return-value"
from enum import Enum
from typing import List, Tuple, Union

from psychopy import core, visual

from bcipy.core.stimuli import resize_image
from bcipy.feedback.feedback import Feedback
from bcipy.helpers.clock import Clock


class FeedbackType(Enum):
    TEXT = 'TEXT'
    IMAGE = 'IMAGE'


class VisualFeedback(Feedback):
    """Visual Feedback."""

    def __init__(self, display: visual.Window, parameters: dict, clock: Clock) -> None:

        # Register Feedback Type
        self.feedback_type = 'Visual Feedback'

        super(VisualFeedback, self).__init__(self.feedback_type)

        # Display Window
        self.display = display

        # Parameters
        self.parameters = parameters
        self.font_stim: str = self.parameters['feedback_font']
        self.height_stim: int = self.parameters['feedback_stim_height']

        pos_x = self.parameters['feedback_pos_x']
        pos_y = self.parameters['feedback_pos_y']

        self.pos_stim = (pos_x, pos_y)

        self.feedback_length = self.parameters['feedback_duration']
        self.color = self.parameters['feedback_color']

        self.clock = clock

        self.feedback_timestamp_label = 'visual_feedback'

    def administer(
            self,
            stimulus: str,
            stimuli_type=FeedbackType.TEXT) -> List[Tuple[str, float]]:
        """Administer.

        Administer visual feedback. Timing information from parameters,
            current feedback given by stimulus.
        """

        stim = self._construct_stimulus(
            stimulus,
            self.pos_stim,
            self.color,
            stimuli_type)

        time = self._show_stimuli(stim)

        core.wait(self.feedback_length)

        return [time]

    def _show_stimuli(self, stimulus: Union[visual.TargetStim, visual.ImageStim]) -> Tuple[str, float]:
        stimulus.draw()
        time = [self.feedback_timestamp_label, self.clock.getTime()]  # TODO: use callback for better precision
        self.display.flip()
        return time

    def _resize_image(
            self,
            stimulus: str,
            display_size: Tuple[float, float],
            stimuli_height: int) -> Tuple[float, float]:
        return resize_image(
            stimulus, display_size, stimuli_height)

    def _construct_stimulus(
            self,
            stimulus: str,
            pos: Tuple[float, float],
            fill_color: str,
            stimuli_type: FeedbackType) -> Union[visual.TargetStim, visual.ImageStim]:
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
