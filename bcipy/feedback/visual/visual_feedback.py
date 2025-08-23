"""Visual feedback module.

This module provides visual feedback functionality for BciPy, implementing
visual-based feedback mechanisms using PsychoPy for stimulus presentation.
"""

# mypy: disable-error-code="return-value"
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from psychopy import core, visual

from bcipy.core.stimuli import resize_image
from bcipy.feedback.feedback import Feedback, FeedbackType, StimuliType
from bcipy.helpers.clock import Clock


class VisualFeedback(Feedback):
    """Visual feedback implementation.

    This class provides visual feedback functionality, allowing for the presentation
    of text and image stimuli with precise timing control.

    Attributes:
        feedback_type (FeedbackType): Type of feedback (VIS).
        display (visual.Window): PsychoPy window for display.
        parameters (Dict[str, Any]): Configuration parameters for feedback.
        font_stim (str): Font to use for text stimuli.
        height_stim (int): Height of stimuli.
        pos_stim (Tuple[float, float]): Position for stimuli presentation.
        feedback_length (float): Duration of feedback presentation.
        color (str): Color of the feedback stimulus.
        clock (Clock): Clock for timing control.
        feedback_timestamp_label (str): Label for feedback timing.
    """

    def __init__(self, display: visual.Window, parameters: Dict[str, Any], clock: Clock) -> None:
        """Initialize Visual Feedback.

        Args:
            display (visual.Window): PsychoPy window for display.
            parameters (Dict[str, Any]): Configuration parameters for feedback.
            clock (Clock): Clock instance for timing control.
        """
        # Register Feedback Type
        self.feedback_type = FeedbackType.VIS

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
            stimuli_type: StimuliType = StimuliType.TEXT) -> List[Tuple[str, float]]:
        """Administer visual feedback.

        Presents visual feedback stimulus and records timing information.

        Args:
            stimulus (str): The stimulus to present (text or image path).
            stimuli_type (StimuliType, optional): Type of stimulus to present.
                Defaults to StimuliType.TEXT.

        Returns:
            List[Tuple[str, float]]: List containing timing information
                in the format [(label, timestamp)].

        Raises:
            ValueError: If an unsupported stimulus type is provided.
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
        """Show the stimulus and record timing.

        Args:
            stimulus (Union[visual.TargetStim, visual.ImageStim]): The stimulus to present.

        Returns:
            Tuple[str, float]: Timing information in the format (label, timestamp).
        """
        stimulus.draw()
        time = [self.feedback_timestamp_label, self.clock.getTime()]
        self.display.flip()
        return time

    def _resize_image(
            self,
            stimulus: str,
            display_size: Tuple[float, float],
            stimuli_height: int) -> Tuple[float, float]:
        """Resize an image stimulus to fit the display.

        Args:
            stimulus (str): Path to the image file.
            display_size (Tuple[float, float]): Size of the display window.
            stimuli_height (int): Desired height of the stimulus.

        Returns:
            Tuple[float, float]: New size of the image (width, height).
        """
        return resize_image(
            stimulus, display_size, stimuli_height)

    def _construct_stimulus(
            self,
            stimulus: str,
            pos: Tuple[float, float],
            fill_color: str,
            stimuli_type: StimuliType) -> Union[visual.TargetStim, visual.ImageStim]:
        """Construct a visual stimulus.

        Creates either a text or image stimulus based on the provided type.

        Args:
            stimulus (str): The stimulus content (text or image path).
            pos (Tuple[float, float]): Position for the stimulus.
            fill_color (str): Color for text stimuli.
            stimuli_type (StimuliType): Type of stimulus to create.

        Returns:
            Union[visual.TargetStim, visual.ImageStim]: The created stimulus object.

        Raises:
            ValueError: If an unsupported stimulus type is provided.
        """
        if stimuli_type == StimuliType.IMAGE:
            image_stim = visual.ImageStim(
                win=self.display,
                image=stimulus,
                mask=None,
                pos=pos,
                ori=0.0)
            image_stim.size = self._resize_image(
                stimulus, self.display.size, self.height_stim)
            return image_stim
        if stimuli_type == StimuliType.TEXT:
            return visual.TextStim(
                win=self.display,
                font=self.font_stim,
                text=stimulus,
                height=self.height_stim,
                pos=pos,
                color=fill_color)
        raise ValueError(
            f'VisualFeedback asked to create a stimulus type=[{stimuli_type}] that is not supported.')
