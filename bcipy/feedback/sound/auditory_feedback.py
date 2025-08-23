"""Auditory feedback module.

This module provides auditory feedback functionality for BciPy, implementing
sound-based feedback mechanisms using sounddevice for audio playback.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import sounddevice as sd
from psychopy import core

from bcipy.feedback.feedback import Feedback, FeedbackType


class AuditoryFeedback(Feedback):
    """Auditory feedback implementation.

    This class provides sound-based feedback functionality, allowing for
    the playback of audio stimuli with precise timing control.

    Attributes:
        feedback_type (FeedbackType): Type of feedback (AUD).
        parameters (Dict[str, Any]): Configuration parameters for feedback.
        sound_buffer_time (float): Buffer time for sound playback.
        feedback_timestamp_label (str): Label for feedback timing.
        clock (core.Clock): Clock for timing control.
    """

    def __init__(self, parameters: Dict[str, Any], clock: core.Clock) -> None:
        """Initialize Auditory Feedback.

        Args:
            parameters (Dict[str, Any]): Configuration parameters for feedback.
            clock (core.Clock): Clock instance for timing control.
        """
        # Register Feedback Type
        self.feedback_type = FeedbackType.AUD

        super(AuditoryFeedback, self).__init__(self.feedback_type)

        # Parameters Dictionary
        self.parameters = parameters
        # this should not be changed. Needed to play sound correctly
        self.sound_buffer_time = 1
        self.feedback_timestamp_label = 'auditory_feedback'

        # Clock
        self.clock = clock

    def administer(self, sound: Union[List[float], List[List[float]]], fs: int,
                   assertion: Optional[Any] = None) -> List[List[Union[str, float]]]:
        """Administer auditory feedback.

        Plays the provided sound and records the timing of the feedback.

        Args:
            sound (Union[List[float], List[List[float]]]): Sound data to play.
            fs (int): Sampling frequency of the sound.
            assertion (Optional[Any]): Optional assertion to check before playing.
                Currently not used.

        Returns:
            List[List[Union[str, float]]]: List containing timing information
                in the format [[label, timestamp]].
        """
        timing = []

        if assertion:
            pass

        time = [self.feedback_timestamp_label, self.clock.getTime()]
        sd.play(sound, fs, blocking=True)
        core.wait(self.sound_buffer_time)
        timing.append(time)

        return timing
