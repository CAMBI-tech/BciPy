"""Base calibration task."""

from typing import Iterator, List, NamedTuple, Optional, Tuple

from psychopy import core, visual

from bcipy.acquisition import ClientManager
from bcipy.config import TRIGGER_FILENAME, WAIT_SCREEN_MESSAGE
from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import (DEFAULT_TEXT_FIXATION, StimuliOrder,
                                   TargetPositions,
                                   generate_calibration_inquiries)
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.task import (get_user_input, pause_calibration,
                                trial_complete_message)
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, convert_timing_triggers,
                                    offset_label)
from bcipy.task import Task


class Inquiry(NamedTuple):
    """Represents a single Inquiry"""
    # TODO: types should also work for VEP
    stimuli: List[str]
    durations: List[float]
    colors: List[str]


class BaseCalibrationTask(Task):
    """Base Calibration Task.

    Calibration task performs a stimulus inquiry to elicit a response. Subclasses
    can define a Display for specializing the nature of the stimulus
    (RSVP, Matrix, etc).

    PARAMETERS:
    ----------
    win (PsychoPy Display)
    daq (Data Acquisition Client)
    parameters (dict)
    file_save (str)
    """

    def __init__(self, win: visual.Window, daq: ClientManager,
                 parameters: Parameters, file_save: str) -> None:
        super().__init__()
        self.window = win

        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = Clock()
        self.symbol_set = alphabet(parameters)

        self.file_save = file_save
        self.trigger_handler = TriggerHandler(self.file_save, TRIGGER_FILENAME,
                                              FlushFrequency.EVERY)

        self.wait_screen_message = WAIT_SCREEN_MESSAGE
        self.wait_screen_message_color = parameters['stim_color']

        self.inquiry_generator = self.init_inquiry_generator()
        self.display = self.init_display()

    @property
    def enable_breaks(self) -> bool:
        """Whether to allow breaks"""
        return self.parameters['enable_breaks']

    def wait(self, seconds: Optional[float] = None) -> None:
        """Pause for a time.

        Parameters
        ----------
        - seconds : duration of time to wait; if missing, defaults to the
        value of the parameter `'task_buffer_length'`
        """
        seconds = seconds or self.parameters['task_buffer_length']
        core.wait(seconds)

    def init_display(self) -> MatrixDisplay:
        """Initialize the display"""
        return init_calibration_display_task(self.parameters, self.window,
                                             self.experiment_clock,
                                             self.symbol_set)

    def init_inquiry_generator(self) -> Iterator[Inquiry]:
        """Initializes a generator that returns inquiries to be presented."""
        parameters = self.parameters
        schedule = generate_calibration_inquiries(
            self.symbol_set,
            inquiry_count=parameters['stim_number'],
            stim_per_inquiry=parameters['stim_length'],
            stim_order=StimuliOrder(parameters['stim_order']),
            jitter=parameters['stim_jitter'],
            target_positions=TargetPositions(parameters['target_positions']),
            percentage_without_target=parameters['nontarget_inquiries'],
            timing=[
                parameters['time_prompt'], parameters['time_fixation'],
                parameters['time_flash']
            ],
            color=[
                parameters['target_color'], parameters['fixation_color'],
                parameters['stim_color']
            ])
        return (Inquiry(*inq) for inq in schedule)

    def trigger_type(self, symbol: str, target: str,
                     index: int) -> TriggerType:
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

    def present_inquiry(self, index: int,
                        inquiry: Inquiry) -> List[Tuple[str, float]]:
        """Present the given inquiry and return the trigger timing info.

        Parameters
        ----------
        - index : current index
        - inquiry : next sequence of stimuli to present.
    
        Returns
        -------
        List of stim_times (tuples representing the stimulus and time that
            it was presented relative to the experiment clock)
        """
        self.display.update_task_bar(str(index + 1))
        self.display.draw_static()
        self.window.flip()

        self.display.schedule_to(inquiry.stimuli, inquiry.durations,
                                 inquiry.colors)
        self.wait()
        timing = self.display.do_inquiry()
        return timing

    def user_wants_to_continue(self, first_inquiry: bool = False) -> bool:
        """Check if user wants to continue or terminate.

        Returns
        -------
        - `True` to continue
        - `False` to finish the task.
        """
        should_continue = get_user_input(self.display,
                                         self.wait_screen_message,
                                         self.wait_screen_message_color,
                                         first_run=first_inquiry)
        if not should_continue:
            self.logger.info('User wants to exit.')
        return should_continue

    def execute(self) -> str:
        """Task run loop."""
        self.logger.info(f'Starting {self.name()}!')
        self.wait()

        inq_index = 0
        while self.user_wants_to_continue(inq_index == 0):
            try:
                inquiry = next(self.inquiry_generator)
            except StopIteration:
                break
            if self.enable_breaks:
                # Take a break at the defined interval
                pause_calibration(self.window, self.display, inq_index,
                                  self.parameters)

            timing = self.present_inquiry(inq_index, inquiry)

            # Write triggers for the inquiry
            self.write_trigger_data(timing, first_run=inq_index == 0)

            # Wait for a time
            self.wait()
            inq_index += 1

        self.exit_display()
        self.cleanup()

        self.write_offset_trigger()
        # TODO: write session data?
        # self.write_stimuli_positions()

        return self.file_save

    def exit_display(self) -> None:
        """Close the UI and cleanup."""
        # Say Goodbye!
        self.display.info_text = trial_complete_message(
            self.window, self.parameters)
        self.display.draw_static()
        self.window.flip()

        # Allow for some additional data to be collected for later processing
        self.wait()

    def cleanup(self) -> None:
        """Any cleanup code to run after the last inquiry is complete."""

    def write_trigger_data(self, timing: List[Tuple[str, float]],
                           first_run) -> None:
        """Write Trigger Data.

        Using the timing provided from the display and calibration information from the data acquisition
        client, write trigger data in the correct format.

        *Note on offsets*: we write the full offset value which can be used to transform all stimuli to the time since
            session start (t = 0) for all values (as opposed to most system clocks which start much higher).
            We do not write the calibration trigger used to generate this offset from the display.
            See MatrixDisplay._trigger_pulse() for more information.
        """
        if first_run:
            triggers = []
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = client.offset(self.display.first_stim_time
                                     ) - self.display.first_stim_time
                triggers.append(Trigger(label, TriggerType.OFFSET, time))
            self.trigger_handler.add_triggers(triggers)

        # make sure triggers are written for the inquiry
        self.trigger_handler.add_triggers(
            convert_timing_triggers(timing, timing[0][0], self.trigger_type))

    def write_offset_trigger(self) -> None:
        """Append an offset value to the end of the trigger file.
        """
        # To help support future refactoring or use of lsl timestamps only
        # we write only the sample offset here.
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix='daq_sample_offset')
            time = client.offset(self.display.first_stim_time)
            triggers.append(Trigger(label, TriggerType.SYSTEM, time))

        self.trigger_handler.add_triggers(triggers)
        self.trigger_handler.close()

    def name(self) -> str:
        return 'Matrix Calibration Task'


def init_calibration_display_task(
        parameters: Parameters, window: visual.Window, experiment_clock: Clock,
        symbol_set: core.StaticPeriod) -> MatrixDisplay:
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(-0.6, 0.4),
                                stim_height=0.1,
                                stim_inquiry=[''] * parameters['stim_length'],
                                stim_colors=[parameters['stim_color']] *
                                parameters['stim_length'],
                                stim_timing=[10] * parameters['stim_length'],
                                is_txt_stim=parameters['is_txt_stim'],
                                prompt_time=parameters["time_prompt"])

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height'],
                                  padding=parameters['task_padding'])

    return MatrixDisplay(window,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         rows=parameters['matrix_rows'],
                         columns=parameters['matrix_columns'],
                         width_pct=parameters['matrix_width'],
                         height_pct=1 - (2 * task_bar.height_pct),
                         trigger_type=parameters['trigger_type'],
                         symbol_set=symbol_set)
