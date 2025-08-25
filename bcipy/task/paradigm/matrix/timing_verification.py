"""Matrix timing verification module.

This module provides functionality for verifying display timing in Matrix tasks
using photodiode stimuli. It alternates between solid and empty boxes that can
be measured with a photodiode to ensure accurate stimulus presentation timing.
"""

from itertools import cycle, islice, repeat
from typing import Iterator, List

from bcipy.core.stimuli import PhotoDiodeStimuli, get_fixation, jittered_timing
from bcipy.task import TaskMode
from bcipy.task.calibration import Inquiry
from bcipy.task.paradigm.matrix.calibration import (MatrixCalibrationTask,
                                                    MatrixDisplay)


class MatrixTimingVerificationCalibration(MatrixCalibrationTask):
    """Matrix Timing Verification Task.

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

    name = 'Matrix Timing Verification'
    mode = TaskMode.TIMING_VERIFICATION

    def init_display(self) -> MatrixDisplay:
        """Initialize the display with transparent background.

        Returns:
            MatrixDisplay: Configured matrix display instance.
        """
        display = super().init_display()
        display.start_opacity = 0.0
        return display

    @property
    def symbol_set(self) -> List[str]:
        """Get symbols used in the calibration.

        Returns:
            List[str]: List of symbols with photodiode stimuli inserted.
        """
        return symbols_with_photodiode_stim(super().symbol_set)

    def init_inquiry_generator(self) -> Iterator[Inquiry]:
        """Initialize the inquiry generator for timing verification.

        The generator alternates between solid and empty boxes with specified
        timing parameters. A fixation point is shown between stimuli.

        Returns:
            Iterator[Inquiry]: Generator yielding timing verification inquiries.
        """
        params = self.parameters

        # alternate between solid and empty boxes
        letters = cycle(PhotoDiodeStimuli.list())
        # advance to solid
        next(letters)
        fixation = get_fixation(is_txt=True)

        inq_len = params['stim_length']
        stimuli = [
            PhotoDiodeStimuli.SOLID.value, fixation, *islice(letters, inq_len)
        ]
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


def symbols_with_photodiode_stim(symbols: List[str]) -> List[str]:
    """Replace central symbols with photodiode stimuli.

    Args:
        symbols: List of symbols to modify.

    Returns:
        List[str]: Modified list with central symbols replaced by photodiode
            stimuli.
    """
    mid = int(len(symbols) / 2)

    def sym_at_index(sym: str, index: int) -> str:
        """Get symbol at given index, replacing central indices with stimuli.

        Args:
            sym: Original symbol.
            index: Position in symbol list.

        Returns:
            str: Original symbol or photodiode stimulus.
        """
        if index == mid:
            return PhotoDiodeStimuli.SOLID.value
        if index == mid + 1:
            return PhotoDiodeStimuli.EMPTY.value
        return sym

    return [sym_at_index(sym, i) for i, sym in enumerate(symbols)]
