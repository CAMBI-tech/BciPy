# -*- coding: utf-8 -*-

from bcipy.bci_tasks.rsvp.calibration import RSVPCalibrationTask
from bcipy.bci_tasks.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.bci_tasks.rsvp.copy_phrase_calibration import RSVPCopyPhraseCalibrationTask
from bcipy.bci_tasks.rsvp.icon_to_icon import RSVPIconToIconTask


def start_task(display_window, daq, task_type, parameters, file_save,
               classifier=None, lmodel=None, fake=True):
    # Determine the mode and exp type: send to the correct task.

    # RSVP
    if task_type['mode'] == 'RSVP':

        # CALIBRATION
        if task_type['exp_type'] == 1:

            # try running the experiment
            try:
                calibration_task = RSVPCalibrationTask(
                    display_window, daq, parameters, file_save, fake)

                calibration_task.execute()
            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

        # COPY PHRASE
        elif task_type['exp_type'] == 2:
            # try running the experiment
            try:
                copy_phrase_task = RSVPCopyPhraseTask(
                    display_window, daq, parameters, file_save, classifier,
                    lmodel=lmodel,
                    fake=fake)
                copy_phrase_task.execute()

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

        # COPY PHRASE CALIBRATION
        if task_type['exp_type'] == 3:
            # try running the experiment
            try:
                copy_phrase_calibration = RSVPCopyPhraseCalibrationTask(
                    display_window, daq, parameters, file_save, fake)

                copy_phrase_calibration.execute()

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

        if task_type['exp_type'] == 4:
            #try running the experiment
            try:
                icon_to_icon = RSVPIconToIconTask(display_window, daq,
                                                  parameters, file_save, fake)

                icon_to_icon.execute()

            #Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

    else:
        raise Exception(
            '%s %s Not implemented yet!' % (
                task_type['mode'], task_type['exp_type']))

    return
