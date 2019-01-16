from bcipy.feedback.visual.level_feedback import LevelFeedback
from bcipy.helpers.stimuli_generation import play_sound, soundfiles
from bcipy.tasks.task import Task
from bcipy.tasks.rsvp.calibration.calibration import RSVPCalibrationTask

from bcipy.helpers.triggers import _write_triggers_from_sequence_calibration
from bcipy.helpers.stimuli_generation import random_rsvp_calibration_seq_gen, get_task_info
from bcipy.helpers.bci_task_related import (
    alphabet, trial_complete_message, get_user_input, pause_calibration)

from psychopy import core
import random


class RSVPInterSequenceFeedbackCalibration(Task):
    """RSVP InterSequenceFeedbackCalibration Task uses inter sequence
        feedback to alert user to their current state in order to increase performance
        in a calibration task.

    Calibration task performs an RSVP stimulus sequence to elicit an ERP.
    Parameters will change how many stim and for how long they present.
    Parameters also change color and text / image inputs and alert sounds.


    Input:
        win (PsychoPy Display Object)
        daq (Data Acquisition Object)
        parameters (Dictionary)
        file_save (String)

    Output:
        file_save (String)
    """
    TASK_NAME = 'RSVP Inter Sequence Feedback Calibration Task'

    def __init__(self, win, daq, parameters, file_save):
        super(RSVPInterSequenceFeedbackCalibration, self).__init__()
        self._task = RSVPCalibrationTask(
            win,
            daq,
            parameters,
            file_save)

        self.daq = daq
        self.alp = self._task.alp
        self.rsvp = self._task.rsvp
        self.parameters = parameters
        self.file_save = file_save
        self.enable_breaks = self._task.enable_breaks
        self.window = self._task.window
        self.num_sti = self._task.num_sti
        self.len_sti = self._task.len_sti
        self.is_txt_sti = self.rsvp.is_txt_sti
        self.stimuli_height = self._task.stimuli_height
        self.color = self._task.color
        self.timing = self._task.timing
        self.wait_screen_message = self._task.wait_screen_message
        self.wait_screen_message_color = self._task.wait_screen_message_color

        self.visual_feedback = LevelFeedback(
            display=self.window,
            parameters=self.parameters,
            clock=self._task.experiment_clock)

        self.feedback_line_color = self.parameters['feedback_line_color']

    def execute(self):
        self.logger.debug(f'Starting {self.name()}!')
        run = True

        # Check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        # Begin the Experiment
        while run:

            # Get random sequence information given stimuli parameters
            (ele_sti, timing_sti,
             color_sti) = random_rsvp_calibration_seq_gen(
                 self.alp,
                 num_sti=self.num_sti,
                 len_sti=self.len_sti,
                 timing=self.timing,
                 is_txt=self.is_txt_sti,
                 color=self.color)

            (task_text, task_color) = get_task_info(self.num_sti,
                                                    self._task.task_info_color)

            # Execute the RSVP sequences
            for idx_o in range(len(task_text)):

                # check user input to make sure we should be going
                if not get_user_input(self.rsvp, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

                if self.enable_breaks:
                    pause_calibration(self.window, self.rsvp, idx_o,
                                      self.parameters)

                # update task state
                self.rsvp.update_task_state(
                    text=task_text[idx_o],
                    color_list=task_color[idx_o])

                # Draw and flip screen
                self.rsvp.draw_static()
                self.window.flip()

                # Get height
                self.rsvp.sti.height = self.stimuli_height

                # Schedule a sequence
                self.rsvp.stim_sequence = ele_sti[idx_o]

                # check if text stimuli or not for color information
                if self.is_txt_sti:
                    self.rsvp.color_list_sti = color_sti[idx_o]

                self.rsvp.time_list_sti = timing_sti[idx_o]

                # Wait for a time
                core.wait(self._task.buffer_val)

                # Do the sequence
                last_sequence_timing = self.rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, self._task.trigger_file)

                # TODO implement feedback decision maker
                position = self._get_feedback_decision()
                timing = self.visual_feedback.administer(position=position)

                # TODO write the visual feedback timing

                # Wait for a time
                core.wait(self._task.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self._task.buffer_val)

        if self.daq.is_calibrated:
            _write_triggers_from_sequence_calibration(
                ['offset', self.daq.offset], self._task.trigger_file, offset=True)

        # Close this sessions trigger file and return some data
        self._task.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self._task.eeg_buffer)

        return self.file_save

    def _get_feedback_decision(self):
        return random.randint(1, 5)

    @classmethod
    def label(cls):
        return RSVPInterSequenceFeedbackCalibration.TASK_NAME

    def name(self):
        return RSVPInterSequenceFeedbackCalibration.TASK_NAME
