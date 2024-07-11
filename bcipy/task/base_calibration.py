"""Base calibration task."""

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

from psychopy import core, visual

import bcipy.task.data as session_data
from bcipy.acquisition import ClientManager
from bcipy.config import (SESSION_DATA_FILENAME, TRIGGER_FILENAME,
                          WAIT_SCREEN_MESSAGE)
from bcipy.display import Display
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import _save_session_related_data
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
    """Represents a single Inquiry.

    stimuli - list of [target, fixation, *inquiry_symbols]
    durations - duration in seconds to display each symbol
    colors - color for each symbol
    """
    stimuli: List[str]
    durations: List[float]
    colors: List[str]

    @property
    def target(self) -> str:
        """target symbol"""
        return self.stimuli[0]


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

    Subclasses should override the provided MODE and can specialize behavior by overriding
    the following methods:
        - init_display ; initializes the stimulus display
        - init_inquiry_generator ; creates a generator that returns inquiries to be presented
        - trigger_type ; used for assigning trigger types to the timing data
        - session_task_data ; provide task-specific session data
        - session_inquiry_data ; provide task-specific inquiry data to the session
        - cleanup ; perform any necessary cleanup (closing connections, etc.)
    """

    MODE = 'Undefined'

    def __init__(self, win: visual.Window, daq: ClientManager,
                 parameters: Parameters, file_save: str) -> None:
        super().__init__()
        self.window = win

        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = Clock()
        self.start_time = self.experiment_clock.getTime()
        self._symbol_set = alphabet(parameters)

        self.file_save = file_save
        self.trigger_handler = TriggerHandler(self.file_save, TRIGGER_FILENAME,
                                              FlushFrequency.EVERY)

        self.wait_screen_message = WAIT_SCREEN_MESSAGE
        self.wait_screen_message_color = parameters['stim_color']

        self.inquiry_generator = self.init_inquiry_generator()
        self.display = self.init_display()

        self.session = self.init_session()
        self.write_session_data()

    @property
    def enable_breaks(self) -> bool:
        """Whether to allow breaks"""
        return self.parameters['enable_breaks']

    @property
    def symbol_set(self) -> List[str]:
        """Symbols used in the calibration"""
        return self._symbol_set

    def wait(self, seconds: Optional[float] = None) -> None:
        """Pause for a time.

        Parameters
        ----------
        - seconds : duration of time to wait; if missing, defaults to the
        value of the parameter `'task_buffer_length'`
        """
        seconds = seconds or self.parameters['task_buffer_length']
        core.wait(seconds)

    @abstractmethod
    def init_display(self) -> Display:
        """Initialize the display"""
        ...

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
        return (Inquiry(*inq) for inq in schedule.inquiries())

    def init_session(self) -> session_data.Session:
        """Initialize the session data."""
        return session_data.Session(save_location=self.file_save,
                                    task='Calibration',
                                    mode=self.MODE,
                                    symbol_set=self.symbol_set,
                                    task_data=self.session_task_data())

    def session_task_data(self) -> Optional[Dict[str, Any]]:
        """"Task-specific session data"""
        return None

    def trigger_type(self, symbol: str, target: str,
                     index: int) -> TriggerType:
        """Trigger Type.

        This method is passed to convert_timing_triggers to properly assign TriggerTypes
            to the timing of stimuli presented. The default implementation assumes an
            inquiry with the shape [prompt, fixation, *symbols].
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
        self.logger.info(f'Starting {self.name}!')
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
            self.write_trigger_data(timing, inq_index == 0)
            self.add_session_data(inquiry)

            # Wait for a time
            self.wait()
            inq_index += 1

        self.exit_display()
        self.write_offset_trigger()
        self.cleanup()

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

    def write_session_data(self) -> None:
        """Save session data to disk."""
        if self.session:
            session_file = _save_session_related_data(
                f"{self.file_save}/{SESSION_DATA_FILENAME}",
                self.session.as_dict())
            session_file.close()

    def stim_labels(self, inquiry: Inquiry) -> List[str]:
        """labels for each stimuli"""
        return [
            str(self.trigger_type(symbol, inquiry.target, index))
            for index, symbol in enumerate(inquiry.stimuli)
        ]

    def add_session_data(self, inquiry: Inquiry) -> None:
        """Adds the latest inquiry to the session data."""
        data = session_data.Inquiry(
            stimuli=inquiry.stimuli,
            timing=inquiry.durations,
            triggers=[],
            target_info=self.stim_labels(inquiry),
            target_letter=inquiry.target,
            task_data=self.session_inquiry_data(inquiry))
        self.session.add_sequence(data)
        self.session.total_time_spent = self.experiment_clock.getTime(
        ) - self.start_time
        self.write_session_data()

    def session_inquiry_data(self, inquiry: Inquiry) -> Optional[Dict[str, Any]]:
        """Defines task-specific session data for each inquiry."""
        return None