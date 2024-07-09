"""Code for constructing and executing registered tasks"""
# mypy: disable-error-code="arg-type, misc"
from typing import List, Optional
from psychopy import visual

from bcipy.task import Task
from bcipy.task.exceptions import TaskRegistryException

from bcipy.acquisition import ClientManager
from bcipy.helpers.parameters import Parameters
from bcipy.signal.model import SignalModel
from bcipy.language import LanguageModel


def make_task(
        display_window: visual.Window,
        daq: ClientManager,
        task: Task,
        parameters: Parameters,
        file_save: str,
        signal_models: Optional[List[SignalModel]] = None,
        language_model: Optional[LanguageModel] = None,
        fake: bool = True) -> Task:
    """Creates a Task based on the provided parameters.

    Parameters:
    -----------
        display_window: psychopy Window
        daq: ClientManager - manages one or more acquisition clients
        task: Task - instance of a Task subclass that will be initialized
        parameters: dict
        file_save: str - path to file in which to save data
        signal_models - list of trained models
        language_model - language model
        fake: boolean - true if eeg stream is randomly generated
    Returns:
    --------
        Task instance
    """

    # TODO: Either standardize the task creation process or add awareness of calibration vs. non-calibration tasks
    try:
        return task(
            display_window,
            daq,
            parameters,
            file_save,
            signal_models,
            language_model,
            fake)
    except Exception as e:
        raise TaskRegistryException(
            f'The task could not be created. Please check the task name and try again. Error: {e}')


def start_task(
        display_window: visual.Window,
        daq: ClientManager,
        task: Task,
        parameters: Parameters,
        file_save: str,
        signal_models: Optional[List[SignalModel]] = None,
        language_model: Optional[LanguageModel] = None,
        fake: bool = True) -> str:
    """Creates a Task and starts execution."""
    bcipy_task = make_task(
        display_window,
        daq,
        task,
        parameters,
        file_save,
        signal_models,
        language_model,
        fake)
    return bcipy_task.execute()
