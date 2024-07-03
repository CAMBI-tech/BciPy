from itertools import cycle
from typing import List
from bcipy.task.paradigm.matrix.calibration import (
    MatrixCalibrationTask, InquirySchedule, DEFAULT_TEXT_FIXATION, MatrixDisplay,
    init_calibration_display_task)
from bcipy.helpers.stimuli import PhotoDiodeStimuli, jittered_timing


class MatrixTimingVerificationCalibration(MatrixCalibrationTask):
    """Matrix Timing Verification Task.

    This task is used for verifying display timing by alternating solid and empty boxes. These
        stimuli can be used with a photodiode to ensure accurate presentations.

    Input:
        win (PsychoPy Display Object)
        daq (Data Acquisition Object)
        parameters (Dictionary)
        file_save (String)

    Output:
        file_save (String)
    """
    name = 'Matrix Timing Verification Task'

    def init_display(self) -> MatrixDisplay:
        """Initialize the display"""
        display = init_calibration_display_task(
            self.parameters, self.window, self.experiment_clock,
            symbols_with_photodiode_stim(self.symbol_set))
        display.start_opacity = 0.0
        return display

    def generate_stimuli(self) -> InquirySchedule:
        """Generates the inquiries to be presented.
        """
        samples, times, colors = [], [], []
        [time_target, time_fixation, time_stim] = self.timing
        stimuli = cycle(PhotoDiodeStimuli.list())
        stim_timing = jittered_timing(time_stim, self.jitter, self.stim_length)

        # advance iterator to start on the solid stim.
        next(stimuli)

        # alternate between solid and empty boxes
        inq_stim = [PhotoDiodeStimuli.SOLID.value, DEFAULT_TEXT_FIXATION
                    ] + [next(stimuli) for _ in range(self.stim_length)]
        inq_times = [time_target, time_fixation
                     ] + stim_timing
        inq_colors = [self.color[0]] * (self.stim_length + 2)

        for _ in range(self.stim_number):
            samples.append(inq_stim)
            times.append(inq_times)
            colors.append(inq_colors)

        return InquirySchedule(samples, times, colors)


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
