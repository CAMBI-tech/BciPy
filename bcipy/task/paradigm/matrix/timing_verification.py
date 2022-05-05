from itertools import cycle
from bcipy.task import Task
from bcipy.task.paradigm.matrix.calibration import MatrixCalibrationTask
from bcipy.helpers.stimuli import PhotoDiodeStimuli


class MatrixTimingVerificationCalibration(Task):
    """Matrix Calibration Task.

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
    TASK_NAME = 'Matrix Timing Verification Task'

    def __init__(self, win, daq, parameters, file_save):
        super(MatrixTimingVerificationCalibration, self).__init__()
        self._task = MatrixCalibrationTask(win, daq, parameters, file_save)
        self.stimuli = PhotoDiodeStimuli.list()
        self.create_testing_grid()
        self._task.matrix.start_opacity = 0
        self._task.matrix.full_grid_opacity = 0
        self._task.matrix.grid_stimuli_height = 0.8
        self._task.generate_stimuli = self.generate_stimuli

    def create_testing_grid(self):
        """Create Testing Grid.

        To test timing, we require the photodiode stimuli to flicker at the rate used in the execute method. The middle
            of the existing grid is replaced with the necessary stimuli.
        """
        mid = int(len(self._task.matrix.symbol_set) / 2)

        for i, stim in enumerate(self.stimuli):
            self._task.matrix.symbol_set[mid + i] = stim

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
        stimuli = cycle(self.stimuli)

        # alternate between solid and empty boxes
        time_stim = self._task.timing
        color_stim = self._task.color

        inq_len = self._task.stim_length
        inq_stim = [next(stimuli) for _ in range(inq_len)]
        inq_times = [time_stim[0] for _ in range(inq_len)]
        inq_colors = [color_stim[0] for _ in range(inq_len)]

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
        return MatrixTimingVerificationCalibration.TASK_NAME

    def name(self):
        return MatrixTimingVerificationCalibration.TASK_NAME
