"""Code for constructing and executing Tasks"""
from bcipy.tasks.rsvp.calibration.alert_tone_calibration import RSVPAlertToneCalibrationTask
from bcipy.tasks.rsvp.calibration.inter_sequence_feedback_calibration import (
    RSVPInterSequenceFeedbackCalibration
)
from bcipy.tasks.rsvp.calibration.calibration import RSVPCalibrationTask
from bcipy.tasks.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.tasks.rsvp.icon_to_icon import RSVPIconToIconTask
from bcipy.tasks.rsvp.calibration.timing_verification import RSVPTimingVerificationCalibration

from bcipy.tasks.task import Task
from bcipy.tasks.exceptions import TaskRegistryException
from bcipy.tasks.task_registry import TaskType


def make_task(display_window, daq, task, parameters, file_save,
              signal_model=None, language_model=None, fake=True,
              auc_filename=None) -> Task:
    """Creates a Task based on the provided parameters.

    Parameters:
    -----------
        display_window: pyschopy Window
        daq: DataAcquisitionClient
        task: TaskType
        parameters: dict
        file_save: str - path to file in which to save data
        signal_model
        language_model - language model
        fake: boolean - true if eeg stream is randomly generated
        auc_filename: str
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

    # ICON TASKS
    if task is TaskType.RSVP_ICON_TO_ICON:
        return RSVPIconToIconTask(display_window, daq,
                                  parameters, file_save, signal_model,
                                  language_model, fake, False, auc_filename)

    if task is TaskType.RSVP_ICON_TO_WORD:
        # pylint: disable=fixme
        # TODO: consider a new class for this scenario.
        return RSVPIconToIconTask(display_window, daq,
                                  parameters, file_save, signal_model,
                                  language_model, fake, True, auc_filename)

    # CALIBRATION FEEDBACK TASKS
    if task is TaskType.RSVP_ALERT_TONE_CALIBRATION:
        return RSVPAlertToneCalibrationTask(
            display_window, daq, parameters, file_save)

    if task is TaskType.RSVP_INTER_SEQUENCE_FEEDBACK_CALIBRATION:
        return RSVPInterSequenceFeedbackCalibration(
            display_window, daq, parameters, file_save)

    if task is TaskType.RSVP_TIMING_VERIFICATION_CALIBRATION:
        return RSVPTimingVerificationCalibration(display_window, daq,
                                                 parameters, file_save)
    raise TaskRegistryException(
        'The provided experiment type is not registered.')


def start_task(display_window, daq, task, parameters, file_save,
               signal_model=None, language_model=None, fake=True, auc_filename=None):
    """Creates a Task and starts execution."""
    task = make_task(display_window, daq, task, parameters, file_save,
                     signal_model, language_model, fake, auc_filename)
    task.execute()
