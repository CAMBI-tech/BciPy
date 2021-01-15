from psychopy import core

from bcipy.display.rsvp.mode.calibration import CalibrationDisplay

from bcipy.tasks.task import Task

from bcipy.helpers.triggers import _write_triggers_from_inquiry_calibration
from bcipy.helpers.stimuli import random_rsvp_calibration_inq_gen, get_task_info
from bcipy.helpers.task import (
    alphabet, trial_complete_message, get_user_input, pause_calibration)


class RSVPCalibrationTask(Task):
    """RSVP Calibration Task.

    Calibration task performs an RSVP stimulus inquiry
        to elicit an ERP. Parameters will change how many stimuli
        and for how long they present. Parameters also change
        color and text / image inputs.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> presenting inquiries -->
        saving data

    PARAMETERS:
    ----------
    win (PsychoPy Display Object)
    daq (Data Acquisition Object)
    parameters (Dictionary)
    file_save (String)
    """

    def __init__(self, win, daq, parameters, file_save):
        super(RSVPCalibrationTask, self).__init__()

        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = parameters['task_buffer_len']
        self.alp = alphabet(parameters)
        self.rsvp = init_calibration_display_task(
            self.parameters, self.window, self.daq,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save
        trigger_save_location = f"{self.file_save}/{parameters['trigger_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w', encoding='utf-8')

        self.wait_screen_message = parameters['wait_screen_message']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']

        self.stim_number = parameters['stim_number']
        self.stim_length = parameters['stim_length']
        self.timing = [parameters['time_target'],
                       parameters['time_cross'],
                       parameters['time_flash']]

        self.color = [parameters['target_color'],
                      parameters['fixation_color'],
                      parameters['stim_color']]

        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['stim_height']

        self.is_txt_stim = parameters['is_txt_stim']
        self.eeg_buffer = parameters['eeg_buffer_len']

        self.enable_breaks = parameters['enable_breaks']

    def generate_stimuli(self):
        """Generates the inquiries to be presented.
        Returns:
        --------
            tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)
        """
        return random_rsvp_calibration_inq_gen(self.alp,
                                               stim_number=self.stim_number,
                                               stim_length=self.stim_length,
                                               timing=self.timing,
                                               is_txt=self.rsvp.is_txt_stim,
                                               color=self.color)

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

            # Get inquiry information given stimuli parameters
            (ele_sti, timing_sti, color_sti) = self.generate_stimuli()

            (task_text, task_color) = get_task_info(self.stim_number,
                                                    self.task_info_color)

            # Execute the RSVP inquiries
            for idx_o in range(len(task_text)):

                # check user input to make sure we should be going
                if not get_user_input(self.rsvp, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

                # Take a break every number of trials defined
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

                # Schedule a inquiry
                self.rsvp.stimuli_inquiry = ele_sti[idx_o]

                # check if text stimuli or not for color information
                if self.is_txt_stim:
                    self.rsvp.stimuli_colors = color_sti[idx_o]

                self.rsvp.stimuli_timing = timing_sti[idx_o]

                # Wait for a time
                core.wait(self.buffer_val)

                # Do the inquiry
                last_inquiry_timing = self.rsvp.do_inquiry()

                # Write triggers for the inquiry
                _write_triggers_from_inquiry_calibration(
                    last_inquiry_timing, self.trigger_file)

                # Wait for a time
                core.wait(self.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.rsvp.text = trial_complete_message(
            self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self.buffer_val)

        if self.daq.is_calibrated:
            _write_triggers_from_inquiry_calibration(
                ['offset', self.daq.offset], self.trigger_file, offset=True)

        # Close this sessions trigger file and return some data
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

        return self.file_save

    def name(self):
        return 'RSVP Calibration Task'


def init_calibration_display_task(
        parameters, window, daq, static_clock, experiment_clock):
    return CalibrationDisplay(
        window,
        static_clock,
        experiment_clock,
        daq.marker_writer,
        info_text=parameters['info_text'],
        info_color=parameters['info_color'],
        info_pos=(parameters['text_pos_x'],
                  parameters['text_pos_y']),
        info_height=parameters['info_height'],
        info_font=parameters['info_font'],
        task_color=[parameters['task_color']],
        task_font=parameters['task_font'],
        task_height=parameters['task_height'],
        stim_font=parameters['stim_font'],
        stim_pos=(parameters['stim_pos_x'],
                  parameters['stim_pos_y']),
        stim_height=parameters['stim_height'],
        stim_colors=[parameters['stim_color'] * 10],
        is_txt_stim=parameters['is_txt_stim'],
        trigger_type=parameters['trigger_type'],
        space_char=parameters['stim_space_char'])
