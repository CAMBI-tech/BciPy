# -*- coding: utf-8 -*-

from bcipy.tasks.rsvp.alert_tone_calibration import RSVPAlertToneCalibrationTask
from bcipy.tasks.rsvp.calibration import RSVPCalibrationTask
from bcipy.tasks.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.tasks.rsvp.copy_phrase_calibration import RSVPCopyPhraseCalibrationTask
from bcipy.tasks.rsvp.icon_to_icon import RSVPIconToIconTask

# TODO
from bcipy.tasks.task import Task

def start_task(display_window, daq, task_type, parameters, file_save,
               classifier=None, lmodel=None, fake=True, auc_filename=None):
    # Determine the mode and exp type: send to the correct task.

    # RSVP
    if task_type['mode'] == 'RSVP':
        task = None
        # CALIBRATION
        if task_type['exp_type'] == 1:
            task = RSVPCalibrationTask(
                display_window, daq, parameters, file_save, fake)

        # COPY PHRASE
        elif task_type['exp_type'] == 2:
            task = RSVPCopyPhraseTask(
                display_window, daq, parameters, file_save, classifier,
                lmodel=lmodel,
                fake=fake)

        # COPY PHRASE CALIBRATION
        elif task_type['exp_type'] == 3:
            task = RSVPCopyPhraseCalibrationTask(
                display_window, daq, parameters, file_save, fake)

        # IconToIcon where icon is not word
        elif task_type['exp_type'] == 4:
            task = RSVPIconToIconTask(display_window, daq,
                                      parameters, file_save, classifier,
                                      lmodel, fake, False, auc_filename)
        # IconToIcon where icon is word
        # TODO: create a different class for the other scenario.
        elif task_type['exp_type'] == 5:
            task = RSVPIconToIconTask(display_window, daq,
                                      parameters, file_save, classifier,
                                      lmodel, fake, True, auc_filename)
        # Alert tone calibration
        elif task_type['exp_type'] == 6:
            task = RSVPAlertToneCalibrationTask(
                display_window, daq, parameters, file_save, fake)

        if not task:
            raise Exception(
                'Experiment type for RSVP not registered in start task')

        task.execute()
    else:
        raise Exception(
            '%s %s Not implemented yet!' % (
                task_type['mode'], task_type['exp_type']))
