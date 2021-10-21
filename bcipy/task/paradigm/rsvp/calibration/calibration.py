from psychopy import core

from bcipy.display.rsvp import (InformationProperties, StimuliProperties,
                                TaskDisplayProperties)
from bcipy.display.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.stimuli import (StimuliOrder, calibration_inquiry_generator,
                                   get_task_info)
from bcipy.helpers.task import (alphabet, get_user_input, pause_calibration,
                                trial_complete_message)
from bcipy.helpers.triggers import _write_triggers_from_inquiry_calibration

from bcipy.task import Task


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
        self.experiment_clock = Clock()
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
        self.stim_order = StimuliOrder(parameters['stim_order'])

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
        return calibration_inquiry_generator(self.alp,
                                             stim_number=self.stim_number,
                                             stim_length=self.stim_length,
                                             stim_order=self.stim_order,
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

                # Schedule a inquiry
                self.rsvp.stimuli_inquiry = ele_sti[idx_o]

                # check if text stimuli or not for color information
                if self.is_txt_stim:
                    self.rsvp.stimuli_colors = color_sti[idx_o]

                self.rsvp.stimuli_timing = timing_sti[idx_o]

                # Wait for a time
                core.wait(self.buffer_val)

                # Do the inquiry
                timing = self.rsvp.do_inquiry()

                # Write triggers for the inquiry
                _write_triggers_from_inquiry_calibration(
                    timing, self.trigger_file)

                # Wait for a time
                core.wait(self.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.rsvp.info_text = trial_complete_message(
            self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self.buffer_val)

        # Write offset
        if self.daq.is_calibrated:
            _write_triggers_from_inquiry_calibration(
                ['offset',
                 self.daq.offset(self.rsvp.first_stim_time)],
                self.trigger_file,
                offset=True)

        # Close this sessions trigger file and return some data
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

        return self.file_save

    def name(self):
        return 'RSVP Calibration Task'


def init_calibration_display_task(
        parameters, window, daq, static_clock, experiment_clock):
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'],
                   parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['info_font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(stim_font=parameters['stim_font'],
                                stim_pos=(parameters['stim_pos_x'],
                                          parameters['stim_pos_y']),
                                stim_height=parameters['stim_height'],
                                stim_inquiry=[''] * parameters['stim_length'],
                                stim_colors=[parameters['stim_color']] * parameters['stim_length'],
                                stim_timing=[10] * parameters['stim_length'],
                                is_txt_stim=parameters['is_txt_stim'])
    task_display = TaskDisplayProperties(
        task_color=[parameters['task_color']],
        task_pos=(-.8, .85),
        task_font=parameters['task_font'],
        task_height=parameters['task_height'],
        task_text=''
    )
    return CalibrationDisplay(
        window,
        static_clock,
        experiment_clock,
        stimuli,
        task_display,
        info,
        trigger_type=parameters['trigger_type'],
        space_char=parameters['stim_space_char'],
        full_screen=parameters['full_screen'])
