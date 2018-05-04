# Calibration Task for RSVP
from psychopy import core

from display.rsvp.rsvp_disp_modes import CalibrationDisplay

from bci_tasks.task import Task

from helpers.triggers import _write_triggers_from_sequence_calibration
from helpers.stim_gen import random_rsvp_calibration_seq_gen, get_task_info
from helpers.bci_task_related import (
    alphabet, trial_complete_message, get_user_input)


class RSVPCalibrationTask(Task):
    """RSVP Calibration Task.

    Calibration task performs an RSVP stimulus sequence
        to elicit an ERP. Parameters will change how many stim
        and for how long they present. Parameters also change
        color and text / image inputs.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> presenting sequences -->
        saving data

    Input:
        win (PsychoPy Display Object)
        daq (Data Acquistion Object)
        parameters (Dictionary)
        file_save (String)
        fake (Boolean)

    Output:
        file_save (String)

    """
    def __init__(self, win, daq, parameters, file_save, fake):
        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = float(parameters['task_buffer_len']['value'])
        self.alp = alphabet(parameters)
        self.rsvp = init_calibration_display_task(
            self.parameters, self.window,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save
        trigger_save_location = self.file_save + '/triggers.txt'
        self.trigger_file = open(trigger_save_location, 'w')

        self.wait_screen_message = parameters['wait_screen_message']['value']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']['value']

        self.num_sti = int(parameters['num_sti']['value'])
        self.len_sti = int(parameters['len_sti']['value'])
        self.timing = [float(parameters['time_target']['value']),
                       float(parameters['time_cross']['value']),
                       float(parameters['time_flash']['value'])]

        self.color = [parameters['target_letter_color']['value'],
                      parameters['fixation_color']['value'],
                      parameters['stimuli_color']['value']]

        self.task_info_color = parameters['task_color']['value']

        self.stimuli_height = float(parameters['sti_height']['value'])

        self.is_txt_sti = True if parameters['is_txt_sti']['value'] == 'true' \
            else False,
        self.eeg_buffer = int(parameters['eeg_buffer_len']['value'])

    def execute(self):
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
                self.alp, num_sti=self.num_sti,
                len_sti=self.len_sti, timing=self.timing,
                is_txt=self.rsvp.is_txt_sti,
                color=self.color)

            (task_text, task_color) = get_task_info(self.num_sti,
                                                    self.task_info_color)


            # Execute the RSVP sequences
            for idx_o in range(len(task_text)):

                # check user input to make sure we should be going
                if not get_user_input(self.rsvp, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

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
                core.wait(self.buffer_val)

                # Do the sequence
                last_sequence_timing = self.rsvp.do_sequence()

                # Write triggers for the sequence
                _write_triggers_from_sequence_calibration(
                    last_sequence_timing, self.trigger_file)

                # Wait for a time
                core.wait(self.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self.buffer_val)

        if self.daq.is_calibrated:
            _write_triggers_from_sequence_calibration(
                ['offset', self.daq.offset], self.trigger_file, offset=True)

        # Close this sessions trigger file and return some data
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

        return self.file_save

    def name(self):
        return 'RSVP Calibration Task'


def init_calibration_display_task(
        parameters, window, static_clock, experiment_clock):
    rsvp = CalibrationDisplay(
        window=window, clock=static_clock,
        experiment_clock=experiment_clock,
        text_info=parameters['text_text']['value'],
        color_info=parameters['color_text']['value'],
        pos_info=(float(parameters['pos_text_x']['value']),
                  float(parameters['pos_text_y']['value'])),
        height_info=float(parameters['txt_height']['value']),
        font_info=parameters['font_text']['value'],
        color_task=['white'],
        font_task=parameters['font_task']['value'],
        height_task=float(parameters['height_task']['value']),
        font_sti=parameters['font_sti']['value'],
        pos_sti=(float(parameters['pos_sti_x']['value']),
                 float(parameters['pos_sti_y']['value'])),
        sti_height=float(parameters['sti_height']['value']),
        stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
        time_list_sti=[3] * 10,
        size_domain_bg=int(parameters['size_domain_bg']['value']),
        color_bg_txt=parameters['color_bg_txt']['value'],
        font_bg_txt=parameters['font_bg_txt']['value'],
        color_bar_bg=parameters['color_bar_bg']['value'],
        is_txt_sti=True if parameters['is_txt_sti']['value'] == 'true' else False,
        trigger_type=parameters['trigger_type']['value'])
    return rsvp
