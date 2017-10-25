from RSVP import *


def start_task(daq, display, task_type, parameters, file_save):

	# Determine the mode and exp type: send to the correct task. 
	if task_type['mode'] is rsvp:
		if task_type['exp_type'] is calibration:
			# try running the experiment
			try:
				trial_data = RSVP.RSVP_calibration_task(daq, display, parameters, file_save)

			# Raise exceptions if any encountered and clean up!!
			except Exception as e:
				raise e

	# The parameters given for task type were incongruent with implemeted works
	else:
		raise Exception('%s %s Not implemented yet!' % (task_type['mode'], task_type['exp_type']))
	return trial_data