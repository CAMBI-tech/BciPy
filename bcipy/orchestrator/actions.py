from bcipy.task import Task
from typing import Optional
import subprocess


class CallbackAction(Task):
    """
    Action for running a callback.
    """
    name = 'Callback Action'

    def __init__(self, callback: callable, *args, **kwargs) -> None:
        super(CallbackAction, self).__init__()
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def execute(self):
        super(CallbackAction, self).execute()
        self.logger.info(f'Executing callback action {self.callback} with args {self.args} and kwargs {self.kwargs}')
        self.callback(*self.args, **self.kwargs)
        self.logger.info(f'Callback action {self.callback} executed')
        return self.name



class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """
    name = 'Code Hook Action'

    def __init__(self, code_hook: str, subprocess=True) -> None:
        super(CodeHookAction, self).__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess

    def execute(self):
        super(CodeHookAction, self).execute()
        if self.subprocess:
            subprocess.Popen(self.code_hook, shell=True)

        else:
            subprocess.run(self.code_hook, shell=True)

        return self.code_hook

class OfflineAnalysisAction(Task):
    """
    Action for running offline analysis.
    """
    name = 'Offline Analysis Action'

    def __init__(self, data_directory: str, parameters_path: Optional[str] = None, alert=True) -> None:
        super(OfflineAnalysisAction, self).__init__()
        self.parameters_path = parameters_path
        self.alert = alert
        self.data_directory = data_directory

    def execute(self):
        super(OfflineAnalysisAction, self).execute()
        cmd = self.construct_command()
        subprocess.Popen(cmd, shell=True)
        return self.data_directory

    def construct_command(self):
        command = 'python -m bcipy.signal.model.offline_analysis'
        command += ' --data_folder ' + self.data_directory
        if self.parameters_path:
            command += ' --parameters_file ' + self.parameters_path
        if self.alert:
            command += ' --alert'
        return command


# Import actual task classes to be mapped to strings
#TODO: This should probably be in `task_registry.py` but it causes a circular import
# A refactor for the TaskType system is probably needed
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
from bcipy.orchestrator.actions import OfflineAnalysisAction
from bcipy.orchestrator.actions import CodeHookAction
from bcipy.orchestrator.actions import CallbackAction

# TODO: Refactor this, and TaskInfo. This is currently redundant.
# While this makes it easier to get the actual task class,
# it is yet another source of truth for the string representation of the
# task. Ideally this would work with fetching subclasses of Task, and 
# string references would align withe the class's name property.
# for now, this makes it easier to initialize tasks and actions from
# the orchestrator.
# TODO: add validation for tasks added to this registry (probably through a class)
task_registry_dict = {
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