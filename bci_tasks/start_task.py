
from RSVP import calibration, copy_phrase, copy_phrase_calibration


def start_task(display_window, task_type, parameters, file_save,
               classifier=None, lmodel=None, fake=True):
    # Determine the mode and exp type: send to the correct task.

    # RSVP
    if task_type['mode'] is 'RSVP':

        # CALIBRATION
        if task_type['exp_type'] is 1:
            # try running the experiment
            try:
                calibration.rsvp_calibration_task(
                    display_window, parameters, file_save, fake)

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

        # COPY PHRASE
        elif task_type['exp_type'] is 2:
            # try running the experiment
            try:
                copy_phrase.rsvp_copy_phrase_task(
                    display_window, parameters, file_save, classifier,
                    lmodel=lmodel,
                    fake=fake)

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

        # COPY PHRASE CALIBRATION
        if task_type['exp_type'] is 3:
            # try running the experiment
            try:
                copy_phrase_calibration \
                    .rsvp_copy_phrase_calibration_task(
                        display_window, parameters, file_save, fake)

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                raise e

    else:
        raise Exception(
            '%s %s Not implemented yet!' % (
                task_type['mode'], task_type['exp_type']))

    return
