from typing import List, Tuple

from psychopy import core

from bcipy.config import TRIGGER_FILENAME, WAIT_SCREEN_MESSAGE
from bcipy.display import InformationProperties, StimuliProperties, TaskDisplayProperties
from bcipy.display.paradigm.matrix.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.stimuli import (StimuliOrder, TargetPositions, calibration_inquiry_generator,
                                   get_task_info)
from bcipy.helpers.task import (alphabet, get_user_input, pause_calibration,
                                trial_complete_message)
from bcipy.helpers.triggers import TriggerHandler, TriggerType, Trigger, FlushFrequency, convert_timing_triggers
from bcipy.task import Task


class MatrixCalibrationTask(Task):
    """Matrix Calibration Task.

    Calibration task performs an Matrix stimulus inquiry
        to elicit an ERP. Parameters change the number of stimuli
        (i.e. the subset of matrix) and for how long they will highlight.
        Parameters also change color and text / image inputs.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> highlighting inquiries -->
        saving data

    PARAMETERS:
    ----------
    win (PsychoPy Display Object)
    daq (Data Acquisition Object)
    parameters (Dictionary)
    file_save (String)
    """

    def __init__(self, win, daq, parameters, file_save):
        super(MatrixCalibrationTask, self).__init__()
        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = Clock()
        self.buffer_val = parameters['task_buffer_length']
        self.alp = alphabet(parameters)

        self.matrix = init_calibration_display_task(
            self.parameters, self.window,
            self.static_clock, self.experiment_clock)

        self.file_save = file_save
        self.trigger_handler = TriggerHandler(
            self.file_save,
            TRIGGER_FILENAME,
            FlushFrequency.EVERY)

        self.wait_screen_message = WAIT_SCREEN_MESSAGE
        self.wait_screen_message_color = parameters['stim_color']

        self.stim_number = parameters['stim_number']
        self.stim_length = parameters['stim_length']
        self.stim_order = StimuliOrder(parameters['stim_order'])
        self.target_positions = TargetPositions(parameters['target_positions'])
        self.nontarget_inquiries = parameters['nontarget_inquiries']

        self.timing = [parameters['time_flash']]
        self.color = [parameters['stim_color']]
        self.task_info_color = parameters['task_color']
        self.stimuli_height = parameters['stim_height']
        self.is_txt_stim = parameters['is_txt_stim']
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
        samples, timing, color = calibration_inquiry_generator(self.alp,
                                                               stim_number=self.stim_number,
                                                               stim_length=self.stim_length,
                                                               stim_order=self.stim_order,
                                                               target_positions=self.target_positions,
                                                               nontarget_inquiries=self.nontarget_inquiries,
                                                               timing=self.timing,
                                                               color=self.color)

        return (samples, timing, color)

    def trigger_type(self, symbol: str, target: str, index: int) -> TriggerType:
        """Trigger Type.

        This method is passed to convert_timing_triggers to properly assign TriggerTypes
            to the timing of stimuli presented.
        """
        if index == 0:
            return TriggerType.PROMPT
        if target == symbol:
            return TriggerType.TARGET
        return TriggerType.NONTARGET

    def execute(self):

        self.logger.info(f'Starting {self.name()}!')
        run = True

        # Check user input to make sure we should be going
        if not get_user_input(self.matrix, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        # Wait for a time
        core.wait(self.buffer_val)

        # Begin the Experiment
        while run:

            # Get inquiry information given stimuli parameters
            (stimuli_labels, stimuli_timing, stimuli_colors) = self.generate_stimuli()

            assert len(stimuli_labels) == len(stimuli_timing)

            (task_text, task_color) = get_task_info(self.stim_number,
                                                    self.task_info_color)

            for inquiry in range(len(task_text)):

                # check user input to make sure we should be going
                if not get_user_input(self.matrix, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

                # Take a break every number of trials defined
                if self.enable_breaks:
                    pause_calibration(self.window, self.matrix, inquiry,
                                      self.parameters)

                # update task state
                self.matrix.update_task_state(
                    text=task_text[inquiry],
                    color_list=task_color[inquiry])

                # Draw and flip screen
                self.matrix.draw_static()
                self.window.flip()

                self.matrix.schedule_to(
                    stimuli_labels[inquiry],
                    stimuli_timing[inquiry],
                    stimuli_colors[inquiry])
                # Schedule a inquiry

                # Wait for a time
                core.wait(self.buffer_val)

                # Do the inquiry
                timing, target = self.matrix.do_inquiry()

                # Write triggers for the inquiry
                self.write_trigger_data(timing, (inquiry == 0))

                # Wait for a time
                core.wait(self.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.matrix.info_text = trial_complete_message(
            self.window, self.parameters)
        self.matrix.draw_static()
        self.window.flip()

        # Allow for some training data to be collected
        core.wait(self.buffer_val)

        self.write_offset_trigger()

        return self.file_save

    def write_trigger_data(self, timing: List[Tuple[str, float]], first_run) -> None:
        """Write Trigger Data.

        Using the timing provided from the display and calibration information from the data acquisition
        client, write trigger data in the correct format.

        *Note on offsets*: we write the full offset value which can be used to transform all stimuli to the time since
            session start (t = 0) for all values (as opposed to most system clocks which start much higher).
            We do not write the calibration trigger used to generate this offset from the display.
            See MatrixDisplay._trigger_pulse() for more information.
        """
        # write offsets. currently, we only check for offsets at the beginning.
        if self.daq.is_calibrated and first_run:
            self.trigger_handler.add_triggers(
                [Trigger(
                    'starting_offset',
                    TriggerType.OFFSET,
                    # offset will factor in true offset and time relative from beginning
                    (self.daq.offset(self.matrix.first_stim_time) - self.matrix.first_stim_time)
                )]
            )

        # make sure triggers are written for the inquiry
        self.trigger_handler.add_triggers(convert_timing_triggers(timing, timing[0][0], self.trigger_type))

    def write_offset_trigger(self) -> None:
        """Append an offset value to the end of the trigger file.
        """
        if self.daq.is_calibrated:
            self.trigger_handler.add_triggers(
                [Trigger(
                    'daq_sample_offset',
                    TriggerType.SYSTEM,
                    # to help support future refactoring or use of lsl timestamps only
                    # we write only the sample offset here
                    self.daq.offset(self.matrix.first_stim_time)
                )])
        self.trigger_handler.close()

    def name(self):
        return 'Matrix Calibration Task'


def init_calibration_display_task(
        parameters, window, static_clock, experiment_clock):
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'],
                   parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(-0.6, 0.4),
                                stim_height=0.1,
                                stim_inquiry=[''] * parameters['stim_length'],
                                stim_colors=[parameters['stim_color']] * parameters['stim_length'],
                                stim_timing=[10] * parameters['stim_length'],
                                is_txt_stim=parameters['is_txt_stim'],
                                prompt_time=parameters["time_prompt"])

    task_display = TaskDisplayProperties(
        task_color=[parameters['task_color']],
        task_pos=(-.8, .85),
        task_font=parameters['font'],
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
