from RSVP import calibration
import pdb


def start_task(daq, display_window, task_type, parameters, file_save):
    try:
        # Determine the mode and exp type: send to the correct task.
        if task_type['mode'] is 'RSVP':
            if task_type['exp_type'] is 1:
                # try running the experiment
                try:
                    trial_data = calibration.RSVP_calibration_task(
                        display_window, daq, parameters, file_save)

                # Raise exceptions if any encountered and clean up!!
                except Exception as e:
                    print "error in start_task"
                    raise e

        # The parameters given for task type were incongruent with implemeted works
        else:
            raise Exception(
                '%s %s Not implemented yet!' % (
                    task_type['mode'], task_type['exp_type']))
        return trial_data

    except Exception as e:
        pdb.set_trace()
