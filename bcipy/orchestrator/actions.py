from task import Task
from typing import Optional
import subprocess

class CallbackAction(Task):
    """
    Action for running a callback.
    """

    def __init__(self, callback: callable, *args, **kwargs) -> None:
        super(CallbackAction, self).__init__()
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
        super(CodeHookAction, self).__init__()
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
        super(OfflineAnalysisAction, self).__init__()
        self.parameters_path = parameters_path
        self.alert = alert
        self.data_directory = data_directory

    def execute(self):
        cmd = self.construct_command()
        subprocess.Popen(cmd, shell=True)
        return self.data_directory
    
    def construct_command(self):
        command = 'python -m bcipy.signal.model.offline_analysis'
        command += ' --data_directory ' + self.data_directory
        if self.parameters_path:
            command += ' --parameters_path ' + self.parameters_path
        if self.alert:
            command += ' --alert'
        return command

    def name(self):
        return 'OfflineAnalysisAction'
