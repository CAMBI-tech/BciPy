from RSVP import calibration


def start_task(daq, display_window, task_type, parameters, file_save):
    # Determine the mode and exp type: send to the correct task.

    # RSVP
    if task_type['mode'] is 'RSVP':

        # CALIBRATION
        if task_type['exp_type'] is 1:
            # try running the experiment
            try:
                trial_data = calibration.rsvp_calibration_task(
                    display_window, daq, parameters, file_save)

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                print "error in start_task for calibration"
                raise e

        # COPY PHRASE
        if task_type['exp_type'] is 2:
            # try running the experiment
            try:
                trial_data = calibration.rsvp_copy_phrase_task(
                    display_window, daq, parameters, file_save)

            # Raise exceptions if any encountered and clean up!!
            except Exception as e:
                print "error in start_task for copy phrase"
                raise e

    # The parameters given for task type were incongruent with
    #   implemeted works
    else:
        raise Exception(
            '%s %s Not implemented yet!' % (
                task_type['mode'], task_type['exp_type']))

    # Return all relevant trial_data
    return trial_data
