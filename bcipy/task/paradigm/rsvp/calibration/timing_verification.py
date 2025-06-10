# mypy: disable-error-code="assignment"
"""RSVP timing verification module.

This module provides functionality for verifying display timing in RSVP tasks
using photodiode stimuli. It alternates between solid and empty boxes that can
be measured with a photodiode to ensure accurate stimulus presentation timing.
"""

from itertools import cycle, islice, repeat
from typing import Any, Iterator, List

from bcipy.core.parameters import Parameters
from bcipy.core.stimuli import PhotoDiodeStimuli, get_fixation, jittered_timing
from bcipy.task import TaskMode
from bcipy.task.calibration import Inquiry
from bcipy.task.paradigm.rsvp.calibration.calibration import \
    RSVPCalibrationTask


class RSVPTimingVerificationCalibration(RSVPCalibrationTask):
    """RSVP Timing Verification Task.

    This task verifies display timing by alternating solid and empty boxes.
    The stimuli can be measured with a photodiode to ensure accurate
    presentation timing.

    Attributes:
        name: Name of the task.
        mode: Task execution mode.
        parameters: Task configuration parameters.
        file_save: Path for saving task data.
        fake: Whether to run in fake (testing) mode.
    """

    name = 'RSVP Timing Verification'
    mode = TaskMode.TIMING_VERIFICATION

    def __init__(self,
                 parameters: Parameters,
                 file_save: str,
                 fake: bool = False,
                 **kwargs: Any) -> None:
        """Initialize the RSVP timing verification task.

        Args:
            parameters: Task configuration parameters.
            file_save: Path for saving task data.
            fake: Whether to run in fake (testing) mode.
            **kwargs: Additional keyword arguments.
        """
        parameters['rsvp_stim_height'] = 0.8
        parameters['rsvp_stim_pos_y'] = 0.0
        super(RSVPTimingVerificationCalibration,
              self).__init__(parameters, file_save, fake=fake)

    @property
    def symbol_set(self) -> List[str]:
        """Get symbols used in the calibration.

        Returns:
            List[str]: List of photodiode stimuli symbols.
        """
        return PhotoDiodeStimuli.list()

    def init_inquiry_generator(self) -> Iterator[Inquiry]:
        """Initialize the inquiry generator for timing verification.

        The generator alternates between solid and empty boxes with specified
        timing parameters. A fixation point is shown between stimuli.

        Returns:
            Iterator[Inquiry]: Generator yielding timing verification inquiries.
        """
        params = self.parameters

        # alternate between solid and empty boxes
        letters = cycle(self.symbol_set)
        target = next(letters)
        fixation = get_fixation(is_txt=True)

        inq_len = params['stim_length']
        stimuli = [target, fixation, *islice(letters, inq_len)]
        durations = [
            params['time_prompt'], params['time_fixation'], *jittered_timing(
                params['time_flash'], params['stim_jitter'], inq_len)
        ]
        colors = [
            params['target_color'], params['fixation_color'],
            *repeat(params['stim_color'], inq_len)
        ]

        return repeat(Inquiry(stimuli, durations, colors),
                      params['stim_number'])
