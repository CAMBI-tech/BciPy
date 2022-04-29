from itertools import cycle
from bcipy.task import Task
from bcipy.task.paradigm.rsvp.calibration.calibration import RSVPCalibrationTask
from bcipy.helpers.stimuli import PhotoDiodeStimuli, get_fixation


class RSVPTimingVerificationCalibration(Task):
    """RSVP Calibration Task.

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
    TASK_NAME = 'RSVP Timing Verification Task'

    def __init__(self, win, daq, parameters, file_save):
        super(RSVPTimingVerificationCalibration, self).__init__()
        parameters['stim_height'] = 0.8
        parameters['stim_pos_y'] = 0.0
        self.stimuli = PhotoDiodeStimuli.list()
        self._task = RSVPCalibrationTask(win, daq, parameters, file_save)
        self._task.generate_stimuli = self.generate_stimuli

    def generate_stimuli(self):
        """Generates the inquiries to be presented.
        Returns:
        --------
            tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)
        """
        samples, times, colors = [], [], []

        # alternate between solid and empty boxes
        letters = cycle(self.stimuli)
        time_prompt, time_fixation, time_stim = self._task.timing
        color_target, color_fixation, color_stim = self._task.color

        target = next(letters)
        fixation = get_fixation(is_txt=True)

        inq_len = self._task.stim_length
        inq_stim = [target, fixation, *[next(letters) for _ in range(inq_len)]]
        inq_times = [time_prompt, time_fixation, *[time_stim for _ in range(inq_len)]]
        inq_colors = [color_target, color_fixation, *[color_stim for _ in range(inq_len)]]

        for _ in range(self._task.stim_number):
            samples.append(inq_stim)
            times.append(inq_times)
            colors.append(inq_colors)

        return (samples, times, colors)

    def execute(self):
        self.logger.debug(f'Starting {self.name()}!')
        self._task.execute()

    @classmethod
    def label(cls):
        return RSVPTimingVerificationCalibration.TASK_NAME

    def name(self):
        return RSVPTimingVerificationCalibration.TASK_NAME
