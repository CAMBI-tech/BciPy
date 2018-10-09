# -*- coding: utf-8 -*-
"""Code for constructing and executing Tasks"""
from bcipy.tasks.rsvp.alert_tone_calibration import RSVPAlertToneCalibrationTask
from bcipy.tasks.rsvp.calibration import RSVPCalibrationTask
from bcipy.tasks.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.tasks.rsvp.copy_phrase_calibration import RSVPCopyPhraseCalibrationTask
from bcipy.tasks.rsvp.icon_to_icon import RSVPIconToIconTask

from bcipy.tasks.task import Task
from bcipy.tasks.task_registry import ExperimentType


def make_task(display_window, daq, task_type, parameters, file_save,
              classifier=None, lmodel=None, fake=True,
              auc_filename=None) -> Task:
    """Creates a Task based on the provided parameters.

    Parameters:
    -----------
        display_window: pyschopy Window
        daq: DataAcquisitionClient
        task_type: dict
        parameters: dict
        file_save: str - path to file in which to save data
        classifier
        lmodel - language model
        fake: boolean - true if eeg stream is randomly generated
        auc_filename: str
    """
    if task_type['mode'] == 'RSVP':
        exp_type = ExperimentType(task_type['exp_type'])

        if exp_type is ExperimentType.RSVP_CALIBRATION:
            return RSVPCalibrationTask(
                display_window, daq, parameters, file_save)

        if exp_type is ExperimentType.RSVP_COPY_PHRASE:
            return RSVPCopyPhraseTask(
                display_window, daq, parameters, file_save, classifier,
                lmodel=lmodel,
                fake=fake)

        if exp_type is ExperimentType.RSVP_COPY_PHRASE_CALIBRATION:
            return RSVPCopyPhraseCalibrationTask(
                display_window, daq, parameters, file_save, fake)

        if exp_type is ExperimentType.RSVP_ICON_TO_ICON:
            return RSVPIconToIconTask(display_window, daq,
                                      parameters, file_save, classifier,
                                      lmodel, fake, False, auc_filename)

        if exp_type is ExperimentType.RSVP_ICON_TO_WORD:
            # pylint: disable=fixme
            # TODO: create a different class for this scenario.
            return RSVPIconToIconTask(display_window, daq,
                                      parameters, file_save, classifier,
                                      lmodel, fake, True, auc_filename)

        if exp_type is ExperimentType.RSVP_ALERT_TONE_CALIBRATION:
            return RSVPAlertToneCalibrationTask(
                display_window, daq, parameters, file_save)

    raise Exception('The provided task is not registered.')


def start_task(display_window, daq, task_type, parameters, file_save,
               classifier=None, lmodel=None, fake=True, auc_filename=None):
    """Creates a Task and starts execution."""
    task = make_task(display_window, daq, task_type, parameters, file_save,
                     classifier, lmodel, fake, auc_filename)
    task.execute()
