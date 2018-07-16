# Calibration Task for RSVP
from psychopy import core

from bcipy.display.rsvp.rsvp_disp_modes import CalibrationDisplay

from bcipy.bci_tasks.task import Task

from bcipy.helpers.triggers import _write_triggers_from_sequence_calibration
from bcipy.helpers.stimuli_generation import random_rsvp_calibration_seq_gen, get_task_info
from bcipy.helpers.bci_task_related import (
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
        self.buffer_val = parameters['task_buffer_len']
        self.alp = alphabet(parameters)
        self.rsvp = init_calibration_display_task(
            self.parameters, self.window, self.daq,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save
        trigger_save_location = f"{self.file_save}/{parameters['triggers_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w')

        self.wait_screen_message = parameters['wait_screen_message']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']

        self.num_sti = parameters['num_sti']
        self.len_sti = parameters['len_sti']
        self.timing = [parameters['time_target'],
                       parameters['time_cross'],
                       parameters['time_flash']]

        self.color = [parameters['target_letter_color'],
                      parameters['fixation_color'],
                      parameters['stimuli_color']]

        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['sti_height']

        self.is_txt_sti = parameters['is_txt_sti']
        self.eeg_buffer = parameters['eeg_buffer_len']

        self.enable_breaks = parameters['enable_breaks']
        self.break_len = parameters['break_len']
        self.break_message = parameters['break_message']
        self.trials_before_break = parameters['trials_before_break']

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

                #Take a break every number of trials defined in parameters.json
                if self.enable_breaks:
                    if not self.trials_before_break == 0:
                        #Check whether any trials have taken place, and, if so,
                        #whether the number of trials performed is divisible
                        #by the number of trials before a break set in parameters
                        if (idx_o != 0) and (idx_o % self.trials_before_break) == 0:
                            #Update countdown every second
                            for counter in range(0,self.break_len):
                                time = self.break_len - counter
                                message = f'{self.break_message} {time}s'
                                self.rsvp.update_task_state(
                                    text=(message),
                                    color_list=task_color[idx_o])
                                self.rsvp.draw_static()
                                self.window.flip()
                                core.wait(1)

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
        parameters, window, daq, static_clock, experiment_clock):
    rsvp = CalibrationDisplay(
        window=window, clock=static_clock,
        experiment_clock=experiment_clock,
        marker_writer=daq.marker_writer,
        text_info=parameters['text_text'],
        color_info=parameters['color_text'],
        pos_info=(parameters['pos_text_x'],
                  parameters['pos_text_y']),
        height_info=parameters['txt_height'],
        font_info=parameters['font_text'],
        color_task=['white'],
        font_task=parameters['font_task'],
        height_task=parameters['height_task'],
        font_sti=parameters['font_sti'],
        pos_sti=(parameters['pos_sti_x'],
                 parameters['pos_sti_y']),
        sti_height=parameters['sti_height'],
        stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
        time_list_sti=[3] * 10,
        size_domain_bg=parameters['size_domain_bg'],
        color_bg_txt=parameters['color_bg_txt'],
        font_bg_txt=parameters['font_bg_txt'],
        color_bar_bg=parameters['color_bar_bg'],
        is_txt_sti=parameters['is_txt_sti'],
        trigger_type=parameters['trigger_type'])
    return rsvp
