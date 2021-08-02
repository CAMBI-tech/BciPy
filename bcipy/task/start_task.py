"""Code for constructing and executing Tasks"""
from bcipy.task.paradigm.rsvp.calibration.alert_tone_calibration import RSVPAlertToneCalibrationTask
from bcipy.task.paradigm.rsvp.calibration.inter_inquiry_feedback_calibration import (
    RSVPInterInquiryFeedbackCalibration
)
from bcipy.task.paradigm.rsvp.calibration.calibration import RSVPCalibrationTask
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.task.paradigm.rsvp.calibration.timing_verification import RSVPTimingVerificationCalibration

from bcipy.task import Task
from bcipy.task.exceptions import TaskRegistryException
from bcipy.task.task_registry import TaskType


def make_task(display_window, daq, task, parameters, file_save,
              signal_model=None, language_model=None, fake=True) -> Task:
    """Creates a Task based on the provided parameters.

    Parameters:
    -----------
        display_window: psychopy Window
        daq: DataAcquisitionClient
        task: TaskType
        parameters: dict
        file_save: str - path to file in which to save data
        signal_model
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
            display_window, daq, parameters, file_save, signal_model,
            language_model, fake=fake)

    # CALIBRATION FEEDBACK TASKS
    if task is TaskType.RSVP_ALERT_TONE_CALIBRATION:
        return RSVPAlertToneCalibrationTask(
            display_window, daq, parameters, file_save)

    if task is TaskType.RSVP_INTER_INQUIRY_FEEDBACK_CALIBRATION:
        return RSVPInterInquiryFeedbackCalibration(
            display_window, daq, parameters, file_save)

    if task is TaskType.RSVP_TIMING_VERIFICATION_CALIBRATION:
        return RSVPTimingVerificationCalibration(display_window, daq,
                                                 parameters, file_save)
    raise TaskRegistryException(
        'The provided experiment type is not registered.')


def start_task(display_window, daq, task, parameters, file_save,
               signal_model=None, language_model=None, fake=True):
    """Creates a Task and starts execution."""
    task = make_task(display_window, daq, task, parameters, file_save,
                     signal_model, language_model, fake)
    task.execute()