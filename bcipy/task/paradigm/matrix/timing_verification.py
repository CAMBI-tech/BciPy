from itertools import cycle, islice, repeat
from typing import Iterator, List

from bcipy.helpers.stimuli import (PhotoDiodeStimuli, get_fixation,
                                   jittered_timing)
from bcipy.task.calibration import Inquiry
from bcipy.task.paradigm.matrix.calibration import (MatrixCalibrationTask,
                                                    MatrixDisplay)


class MatrixTimingVerificationCalibration(MatrixCalibrationTask):
    """Matrix Timing Verification Task.

    This task is used for verifying display timing by alternating solid and empty boxes. These
        stimuli can be used with a photodiode to ensure accurate presentations.

    Input:
        parameters (Dictionary)
        file_save (String)
        fake (Boolean)

    Output:
        file_save (String)
    """
    name = 'Matrix Timing Verification'

    def init_display(self) -> MatrixDisplay:
        """Initialize the display"""
        display = super().init_display()
        display.start_opacity = 0.0
        return display

    @property
    def symbol_set(self) -> List[str]:
        """Symbols used in the calibration"""
        return symbols_with_photodiode_stim(super().symbol_set)

    def init_inquiry_generator(self) -> Iterator[Inquiry]:
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
    """Stim symbols with the central letters swapped out for Photodiode stim.
    """
    mid = int(len(symbols) / 2)

    def sym_at_index(sym, index) -> str:
        if index == mid:
            return PhotoDiodeStimuli.SOLID.value
        if index == mid + 1:
            return PhotoDiodeStimuli.EMPTY.value
        return sym

    return [sym_at_index(sym, i) for i, sym in enumerate(symbols)]
