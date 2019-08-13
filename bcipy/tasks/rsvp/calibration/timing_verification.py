from itertools import cycle
from bcipy.tasks.task import Task
from bcipy.tasks.rsvp.calibration.calibration import RSVPCalibrationTask


class RSVPTimingVerificationCalibration(Task):
    """RSVP Calibration Task that for verifying timing.

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
        self._task = RSVPCalibrationTask(win, daq, parameters, file_save)
        self._task.alp = ["_", "\u2013", "\u2012", "\u25a1"]

    def generate_stimuli(self):
        """Generates the sequences to be presented.
        Returns:
        --------
            tuple(
                samples[list[list[str]]]: list of sequences
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)
        """
        # return self.alp, stim_number=self.stim_number, stim_length=self.stim_length, timing=self.timing

        samples, times, colors = [], [], []
        target = ''
        # alternate between empty box and solid box
        letters = cycle(["\u2013", "\u2012"])

        choices = [next(letters) for _ in range(self.stim_number)]
        
        items = [target, '+', *choices]
            rand_smp = np.random.permutation(rand_smp)
            sample += [alp[i] for i in rand_smp]
            samples.append(sample)
            times.append([timing[i] for i in range(len(timing) - 1)] +
                         [timing[-1]] * stim_length)
            colors.append([color[i] for i in range(len(color) - 1)] +
                          [color[-1]] * stim_length)
    
        return (samples, times, colors)

                                               
    def execute(self):
        self.logger.debug(f'Starting {self.name()}!')
        self._task.execute()

    @classmethod
    def label(cls):
        return RSVPTimingVerificationCalibration.TASK_NAME

    def name(self):
        return RSVPTimingVerificationCalibration.TASK_NAME
