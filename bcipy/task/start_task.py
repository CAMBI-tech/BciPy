"""Code for constructing and executing registered tasks"""
# mypy: disable-error-code="arg-type, misc"
from typing import List, Optional

from psychopy import visual

from bcipy.acquisition import ClientManager
from bcipy.helpers.parameters import Parameters
from bcipy.language import LanguageModel
from bcipy.signal.model import SignalModel
from bcipy.task import Task
from bcipy.task.exceptions import TaskRegistryException
from bcipy.task.paradigm.matrix.calibration import MatrixCalibrationTask
from bcipy.task.paradigm.matrix.copy_phrase import MatrixCopyPhraseTask
from bcipy.task.paradigm.matrix.timing_verification import \
    MatrixTimingVerificationCalibration
from bcipy.task.paradigm.rsvp.calibration.calibration import \
    RSVPCalibrationTask
from bcipy.task.paradigm.rsvp.calibration.inter_inquiry_feedback_calibration import \
    RSVPInterInquiryFeedbackCalibration
from bcipy.task.paradigm.rsvp.calibration.timing_verification import \
    RSVPTimingVerificationCalibration
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.task.paradigm.vep.calibration import VEPCalibrationTask
from bcipy.task.task_registry import TaskType


def make_task(
        display_window: visual.Window,
        daq: ClientManager,
        task: TaskType,
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
        task: TaskType
        parameters: dict
        file_save: str - path to file in which to save data
        signal_models - list of trained models
        language_model - language model
        fake: boolean - true if eeg stream is randomly generated
    Returns:
    --------
        Task instance
    """

    # NORMAL RSVP MODES
    if task is TaskType.RSVP_CALIBRATION:
        return RSVPCalibrationTask(
            display_window, daq, parameters, file_save)

    if task is TaskType.RSVP_COPY_PHRASE:
        return RSVPCopyPhraseTask(
            display_window, daq, parameters, file_save, signal_models,
            language_model, fake=fake)

    if task is TaskType.RSVP_TIMING_VERIFICATION_CALIBRATION:
        return RSVPTimingVerificationCalibration(display_window, daq, parameters, file_save)

    if task is TaskType.RSVP_INTER_INQUIRY_FEEDBACK_CALIBRATION:
        return RSVPInterInquiryFeedbackCalibration(display_window, daq, parameters, file_save)

    if task is TaskType.MATRIX_CALIBRATION:
        return MatrixCalibrationTask(
            display_window, daq, parameters, file_save
        )

    if task is TaskType.MATRIX_TIMING_VERIFICATION_CALIBRATION:
        return MatrixTimingVerificationCalibration(display_window, daq, parameters, file_save)

    if task is TaskType.MATRIX_COPY_PHRASE:
        return MatrixCopyPhraseTask(
            display_window, daq, parameters, file_save, signal_models,
            language_model, fake=fake)

    if task is TaskType.VEP_CALIBRATION:
        return VEPCalibrationTask(display_window, daq, parameters, file_save)

    raise TaskRegistryException(
        'The provided experiment type is not registered.')


def start_task(
        display_window: visual.Window,
        daq: ClientManager,
        task: TaskType,
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
