# mypy: disable-error-code="assignment"
from itertools import cycle, islice, repeat
from typing import Iterator, List

from psychopy import visual

from bcipy.acquisition import ClientManager
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import (PhotoDiodeStimuli, get_fixation,
                                   jittered_timing)
from bcipy.task.base_calibration import Inquiry
from bcipy.task.paradigm.rsvp.calibration.calibration import \
    RSVPCalibrationTask


class RSVPTimingVerificationCalibration(RSVPCalibrationTask):
    """RSVP Calibration Task.

    This task is used for verifying display timing by alternating solid and empty boxes. These
        stimuli can be used with a photodiode to ensure accurate presentations.

    Input:
        win (PsychoPy Window)
        daq (ClientManager)
        parameters (Parameters)
        file_save (str)

    Output:
        file_save (str)
    """
    name = 'RSVP Timing Verification'

    def __init__(self, win: visual.Window, daq: ClientManager,
                 parameters: Parameters, file_save: str) -> None:
        parameters['stim_height'] = 0.8
        parameters['stim_pos_y'] = 0.0
        super(RSVPTimingVerificationCalibration,
              self).__init__(win, daq, parameters, file_save)

    @property
    def symbol_set(self) -> List[str]:
        """Symbols used in the calibration"""
        return PhotoDiodeStimuli.list()

    def init_inquiry_generator(self) -> Iterator[Inquiry]:
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

    @classmethod
    def label(cls) -> str:
        return RSVPTimingVerificationCalibration.TASK_NAME

    def name(self) -> str:
        return RSVPTimingVerificationCalibration.TASK_NAME
