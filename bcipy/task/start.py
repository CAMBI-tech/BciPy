"""Code for constructing and executing registered tasks"""
# mypy: disable-error-code="arg-type, misc"
from typing import List, Optional, Type

from bcipy.task import Task, TaskData
from bcipy.task.paradigm.matrix.copy_phrase import MatrixCopyPhraseTask
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.signal.model import SignalModel
from bcipy.language import LanguageModel


def make_task(
        task: Type[Task],
        parameters: Parameters,
        file_save: str,
        signal_models: Optional[List[SignalModel]] = None,
        language_model: Optional[LanguageModel] = None,
        fake: bool = True) -> Task:
    """Creates a Task based on the provided parameters.

    Parameters:
    -----------
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

    from bcipy.task.calibration import BaseCalibrationTask
    if issubclass(task, BaseCalibrationTask):
        return task(parameters, file_save, fake)

    if task is RSVPCopyPhraseTask:
        return RSVPCopyPhraseTask(
            parameters, file_save, signal_models,
            language_model, fake=fake)

    if task is MatrixCopyPhraseTask:
        return MatrixCopyPhraseTask(
            parameters, file_save, signal_models,
            language_model, fake=fake)

    raise BciPyCoreException('The provided task is not available in main')


def start_task(
        task: Type[Task],
        parameters: Parameters,
        file_save: str,
        signal_models: Optional[List[SignalModel]] = None,
        language_model: Optional[LanguageModel] = None,
        fake: bool = True) -> TaskData:
    """Creates a Task and starts execution."""
    bcipy_task = make_task(
        task,
        parameters,
        file_save,
        signal_models,
        language_model,
        fake)
    return bcipy_task.execute()
