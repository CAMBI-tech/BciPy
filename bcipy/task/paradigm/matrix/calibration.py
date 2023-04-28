from typing import List, Tuple

from psychopy import core

from bcipy.config import TRIGGER_FILENAME, WAIT_SCREEN_MESSAGE
from bcipy.display import Display, InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.stimuli import (DEFAULT_TEXT_FIXATION, StimuliOrder,
                                   TargetPositions,
                                   calibration_inquiry_generator,
                                   InquirySchedule)
from bcipy.helpers.task import (get_user_input, pause_calibration,
                                trial_complete_message)
from bcipy.helpers.triggers import TriggerHandler, TriggerType, Trigger, FlushFrequency, convert_timing_triggers
from bcipy.task import Task
from bcipy.helpers.symbols import alphabet


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
        self.symbol_set = alphabet(parameters)

        self.file_save = file_save
        self.trigger_handler = TriggerHandler(
            self.file_save,
            TRIGGER_FILENAME,
            FlushFrequency.EVERY)

        self.wait_screen_message = WAIT_SCREEN_MESSAGE
        self.wait_screen_message_color = parameters['stim_color']

        self.stim_number = parameters['stim_number']
        self.stim_length = parameters['stim_length']
        self.jitter = parameters['stim_jitter']
        self.stim_order = StimuliOrder(parameters['stim_order'])
        self.target_positions = TargetPositions(parameters['target_positions'])
        self.nontarget_inquiries = parameters['nontarget_inquiries']

        self.timing = [
            parameters['time_prompt'], parameters['time_fixation'],
            parameters['time_flash']
        ]
        self.color = [
            parameters['target_color'], parameters['fixation_color'],
            parameters['stim_color']
        ]
        self.task_info_color = parameters['task_color']
        self.stimuli_height = parameters['stim_height']
        self.is_txt_stim = parameters['is_txt_stim']
        self.enable_breaks = parameters['enable_breaks']

        self.matrix = self.init_display()

    def init_display(self) -> Display:
        """Initialize the display"""
        return init_calibration_display_task(self.parameters, self.window,
                                             self.experiment_clock,
                                             self.symbol_set)

    def generate_stimuli(self) -> InquirySchedule:
        """Generates the inquiries to be presented.
        Returns:
        --------
            tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)
        """
        return calibration_inquiry_generator(
            self.symbol_set,
            stim_number=self.stim_number,
            stim_length=self.stim_length,
            stim_order=self.stim_order,
            jitter=self.jitter,
            target_positions=self.target_positions,
            nontarget_inquiries=self.nontarget_inquiries,
            timing=self.timing,
            color=self.color)

    def trigger_type(self, symbol: str, target: str, index: int) -> TriggerType:
        """Trigger Type.

        This method is passed to convert_timing_triggers to properly assign TriggerTypes
            to the timing of stimuli presented.
        """
        if index == 0:
            return TriggerType.PROMPT
        if symbol == DEFAULT_TEXT_FIXATION:
            return TriggerType.FIXATION
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

            for inquiry in range(self.stim_number):

                # check user input to make sure we should be going
                if not get_user_input(self.matrix, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

                # Take a break every number of trials defined
                if self.enable_breaks:
                    pause_calibration(self.window, self.matrix, inquiry,
                                      self.parameters)

                # update task state
                self.matrix.update_task_bar(str(inquiry + 1))

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
                timing = self.matrix.do_inquiry()

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
        parameters, window, experiment_clock, symbol_set):
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

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height'])

    return MatrixDisplay(
        window,
        experiment_clock,
        stimuli,
        task_bar,
        info,
        trigger_type=parameters['trigger_type'],
        symbol_set=symbol_set)
