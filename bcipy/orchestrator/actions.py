from bcipy.task import Task
from typing import Optional, Callable, Dict
import subprocess
from bcipy.task.paradigm.matrix.calibration import MatrixCalibrationTask
from bcipy.task.paradigm.matrix.copy_phrase import MatrixCopyPhraseTask
from bcipy.task.paradigm.matrix.timing_verification import \
    MatrixTimingVerificationCalibration
from bcipy.task.paradigm.rsvp.calibration.calibration import \
    RSVPCalibrationTask
from bcipy.task.paradigm.rsvp.calibration.timing_verification import \
    RSVPTimingVerificationCalibration
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.task.paradigm.vep.calibration import VEPCalibrationTask


class CallbackAction(Task):
    """
    Action for running a callback.
    """

    def __init__(self, callback: Callable, *args, **kwargs) -> None:
        super().__init__()
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def execute(self):
        self.logger.info(f'Executing callback action {self.callback} with args {self.args} and kwargs {self.kwargs}')
        self.callback(*self.args, **self.kwargs)
        self.logger.info(f'Callback action {self.callback} executed')
        return self.name

    @property
    def name(self):
        return 'CallbackAction'


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """

    def __init__(self, code_hook: str, subprocess=True) -> None:
        super().__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess

    def execute(self):
        if self.subprocess:
            subprocess.Popen(self.code_hook, shell=True)

        else:
            subprocess.run(self.code_hook, shell=True)

        return self.code_hook

    @property
    def name(self):
        return 'CodeHookAction'


class OfflineAnalysisAction(Task):
    """
    Action for running offline analysis.
    """

    def __init__(self, data_directory: str, parameters_path: Optional[str] = None, alert=True) -> None:
        super().__init__()
        self.parameters_path = parameters_path
        self.alert = alert
        self.data_directory = data_directory
        self.command = self.construct_command()

    def execute(self):
        subprocess.Popen(self.command, shell=True)
        return self.data_directory

    def construct_command(self):
        command = 'python -m bcipy.signal.model.offline_analysis'
        command += ' --data_folder ' + self.data_directory
        if self.parameters_path:
            command += ' --parameters_file ' + self.parameters_path
        if self.alert:
            command += ' --alert'
        return command

    @property
    def name(self):
        return 'OfflineAnalysisAction'


task_registry_dict: Dict[str, type] = {
    # Tasks
    'RSVP Calibration': RSVPCalibrationTask,
    'RSVP Copy Phrase': RSVPCopyPhraseTask,
    'RSVP Time Test Calibration': RSVPTimingVerificationCalibration,
    'Matrix Calibration': MatrixCalibrationTask,
    'Matrix Time Test Calibration': MatrixTimingVerificationCalibration,
    'Matrix Copy Phrase': MatrixCopyPhraseTask,
    'VEP Calibration': VEPCalibrationTask,

    # Actions
    'Offline Analysis Action': OfflineAnalysisAction,
    'Code Hook Action': CodeHookAction,
    'Callback Action': CallbackAction
}
