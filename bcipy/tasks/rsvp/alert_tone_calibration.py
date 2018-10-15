"""Calibration Task that uses alert tones to help the user retain focus."""

from bcipy.helpers.stimuli_generation import play_sound, soundfiles
from bcipy.tasks.task import Task
from bcipy.tasks.rsvp.calibration import RSVPCalibrationTask


class RSVPAlertToneCalibrationTask(Task):
    """RSVP Calibration Task that uses alert tones to maintain user focus.

    Calibration task performs an RSVP stimulus sequence to elicit an ERP.
    Parameters will change how many stim and for how long they present.
    Parameters also change color and text / image inputs and alert sounds.

    This task uses the 'alert_sounds_path' parameter to determine which sounds
    to play when a letter is presented to the participant. Sounds are read in
    from the configured directory and cycled through in alphabetical order.
    Sounds with the word 'blank' in the file name will not be played, allowing
    experiments to be setup in which some letters do not have alerts.

    Input:
        win (PsychoPy Display Object)
        daq (Data Acquisition Object)
        parameters (Dictionary)
        file_save (String)

    Output:
        file_save (String)
    """
    TASK_NAME = 'RSVP Alert Tone Calibration Task'

    def __init__(self, win, daq, parameters, file_save):
        super(RSVPAlertToneCalibrationTask, self).__init__()
        self._task = RSVPCalibrationTask(win, daq, parameters, file_save)

        alerts = soundfiles(parameters['alert_sounds_path'])

        def play_sound_callback(_sti):
            sound_path = next(alerts)
            if "blank" in sound_path:
                return
            play_sound(sound_path, sound_load_buffer_time=0,
                       sound_post_buffer_time=0)
        self._task.rsvp.first_stim_callback = play_sound_callback

    def execute(self):
        self.logger.debug(f'Starting {self.name()}!')
        self._task.execute()

    @classmethod
    def label(cls):
        return RSVPAlertToneCalibrationTask.TASK_NAME

    def name(self):
        return RSVPAlertToneCalibrationTask.TASK_NAME
