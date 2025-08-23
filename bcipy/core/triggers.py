"""Trigger utilities for BciPy core.

This module provides classes and functions for managing triggers and calibration events in BciPy experiments.
"""

import logging
import os
from enum import Enum
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from psychopy import core, visual

from bcipy.config import DEFAULT_ENCODING, SESSION_LOG_FILENAME
from bcipy.core.parameters import Parameters
from bcipy.core.stimuli import resize_image
from bcipy.exceptions import BciPyCoreException
from bcipy.helpers.clock import Clock

log = logging.getLogger(SESSION_LOG_FILENAME)

NONE_VALUES = ['0', '0.0']

CALIBRATION_IMAGE_PATH = 'bcipy/static/images/testing/white.png'
MOCK_TRIGGER_DATA = '''starting_offset offset -3635340.451487373
P prompt 3635343.155564679
+ fixation 3635344.159978495
O nontarget 3635344.663714144
Z nontarget 3635344.917679911
B nontarget 3635345.172036632
N nontarget 3635345.425718764
K nontarget 3635345.679572615
U nontarget 3635345.932683132
J nontarget 3635346.186251469
P target 3635346.439412578
_ nontarget 3635346.692164597
T nontarget 3635346.946281828
daq_sample_offset system 293.6437802789733
'''


class CalibrationType(Enum):
    """Calibration Type.

    Enum to define the supported calibration trigger types.
    """
    TEXT = 'text'
    IMAGE = 'image'

    @classmethod
    def list(cls) -> List[str]:
        """Returns all enum values as a list.

        Returns:
            List[str]: List of all enum values.
        """
        return list(map(lambda c: c.value, cls))


class TriggerCallback:
    """Callback handler for trigger events.

    Attributes:
        timing (Optional[Tuple[str, float]]): Timing information for the trigger.
        first_time (bool): Flag indicating if this is the first trigger.
    """
    timing: Optional[Tuple[str, float]] = None
    first_time: bool = True

    def callback(self, clock: Clock, stimuli: str) -> None:
        """Callback function for trigger events.

        Args:
            clock (Clock): Clock instance for timing.
            stimuli (str): Stimulus identifier.
        """
        if self.first_time:
            self.timing = (stimuli, clock.getTime())
            self.first_time = False

    def reset(self) -> None:
        """Reset the callback state."""
        self.timing = None
        self.first_time = True


def _calibration_trigger(
    experiment_clock: Clock,
    trigger_type: str = CalibrationType.TEXT.value,
    trigger_name: str = 'calibration',
    trigger_time: float = 1,
    display: Optional[visual.Window] = None,
    on_trigger: Optional[Callable[[str], None]] = None
) -> Tuple[str, float]:
    """Calibration Trigger.

    Outputs triggers for the purpose of calibrating data and stimuli.
    This is an ongoing difficulty between OS, DAQ devices and stimuli type. This
    code aims to operationalize the approach to finding the correct DAQ samples in
    relation to our trigger code.

    Args:
        experiment_clock (Clock): Clock with getTime() method, which is used in the code
            to report timing of stimuli.
        trigger_type (str): Type of trigger that is desired (text, image, etc).
        trigger_name (str): Name of the trigger used for callbacks / labeling.
        trigger_time (float): Time to display the trigger. Can also be used as a buffer.
        display (Optional[visual.Window]): A window that can display stimuli. Currently, a Psychopy window.
        on_trigger (Optional[Callable[[str], None]]): Optional callback; if present gets called
            when the calibration trigger is fired; accepts a single parameter for the timing information.

    Returns:
        Tuple[str, float]: Timing values for the calibration triggers to be written to trigger file or
            used to calculate offsets.

    Raises:
        BciPyCoreException: If trigger type is invalid or display is required but not provided.
    """
    trigger_callback = TriggerCallback()

    # catch invalid trigger types
    if trigger_type not in CalibrationType.list():
        msg = f'Trigger type=[{trigger_type}] not implemented'
        log.exception(msg)
        raise BciPyCoreException(msg)

    if not display:
        msg = f'Calibration type=[{trigger_type}] requires a display'
        log.exception(msg)
        raise BciPyCoreException(msg)

    if trigger_type == CalibrationType.IMAGE.value:
        calibration_box = visual.ImageStim(
            win=display,
            image=CALIBRATION_IMAGE_PATH,
            pos=(-.5, -.5),
            mask=None,
            ori=0.0)
        calibration_box.size = resize_image(
            CALIBRATION_IMAGE_PATH, display.size, 0.75)

        display.callOnFlip(trigger_callback.callback,
                           experiment_clock, trigger_name)
        if on_trigger is not None:
            display.callOnFlip(on_trigger, trigger_name)

        calibration_box.draw()
        display.flip()

    elif trigger_type == CalibrationType.TEXT.value:
        calibration_text = visual.TextStim(display, text='')

        display.callOnFlip(trigger_callback.callback, experiment_clock,
                           trigger_name)
        if on_trigger is not None:
            display.callOnFlip(on_trigger, trigger_name)

        calibration_text.draw()
        display.flip()

    core.wait(trigger_time)
    if trigger_callback.timing is None:
        log.warning(f'No trigger found for [{trigger_name}]')
        return trigger_name, 0.0
    return trigger_callback.timing


def trigger_durations(params: Parameters) -> Dict[str, float]:
    """Get duration for each type of trigger given in seconds.

    Args:
        params (Parameters): Parameters containing timing information.

    Returns:
        Dict[str, float]: Dictionary mapping trigger types to their durations in seconds.
    """
    return {
        'offset': 0.0,
        'preview': params['preview_inquiry_length'],
        'fixation': params['time_fixation'],
        'prompt': params['time_prompt'],
        'nontarget': params['time_flash'],
        'target': params['time_flash']
    }


class TriggerType(Enum):
    """Enum for the primary types of Triggers."""

    NONTARGET = "nontarget"
    TARGET = "target"
    FIXATION = "fixation"
    PROMPT = "prompt"
    SYSTEM = "system"
    OFFSET = "offset"
    EVENT = "event"
    PREVIEW = "preview"
    ARTIFACT = "artifact"

    @classmethod
    def list(cls) -> List[str]:
        """Returns all enum values as a list.

        Returns:
            List[str]: List of all enum values.
        """
        return list(map(lambda c: c.value, cls))

    @classmethod
    def pre_fixation(cls) -> List['TriggerType']:
        """Returns the subset of TriggerTypes that occur before and including
        the FIXATION trigger.

        Returns:
            List[TriggerType]: List of trigger types that occur before fixation.
        """
        return [
            TriggerType.FIXATION, TriggerType.PROMPT, TriggerType.SYSTEM,
            TriggerType.OFFSET
        ]

    def __str__(self) -> str:
        """String representation of the trigger type.

        Returns:
            str: String representation of the trigger type.
        """
        return f'{self.value}'


class Trigger(NamedTuple):
    """Object that encompasses data for a single trigger instance.

    Attributes:
        label (str): Label for the trigger.
        type (TriggerType): Type of the trigger.
        time (float): Timestamp of the trigger.
    """

    label: str
    type: TriggerType
    time: float

    def __repr__(self) -> str:
        """String representation of the trigger.

        Returns:
            str: String representation of the trigger.
        """
        return f'Trigger: label=[{self.label}] type=[{self.type}] time=[{self.time}]'

    def with_offset(self, offset: float) -> 'Trigger':
        """Construct a copy of this Trigger with the offset adjusted.

        Args:
            offset (float): Offset to apply to the trigger time.

        Returns:
            Trigger: New trigger instance with adjusted time.
        """
        return Trigger(self.label, self.type, self.time + offset)

    @classmethod
    def from_list(cls, lst: List[str]) -> 'Trigger':
        """Constructs a Trigger from a serialized representation.

        Args:
            lst (List[str]): Serialized representation [label, type, stamp].

        Returns:
            Trigger: New trigger instance.

        Raises:
            AssertionError: If input list does not have exactly 3 elements.
        """
        assert len(lst) == 3, "Input must have a label, type, and stamp"
        return cls(lst[0], TriggerType(lst[1]), float(lst[2]))


class FlushFrequency(Enum):
    """Enum that defines how often list of Triggers will be written and dumped."""

    EVERY = "flush after every trigger addition"
    END = "flush at end of session"


def read_data(lines: Iterable[str]) -> List[Trigger]:
    """Read raw trigger data from the given source.

    Args:
        lines (Iterable[str]): Iterable object where each item is a str with data for a single
            trigger.

    Returns:
        List[Trigger]: List of all Triggers in the data.

    Raises:
        BciPyCoreException: If there is an error reading a trigger from any line.
    """
    triggers = []
    for i, line in enumerate(lines):
        try:
            trg = Trigger.from_list(line.split())
            triggers.append(trg)
        except (AssertionError, ValueError) as trg_error:
            raise BciPyCoreException(
                f'Error reading trigger from line {i+1}: {trg_error}'
            ) from trg_error
    return triggers


def offset_label(device_type: Optional[str] = None, prefix: str = 'starting_offset') -> str:
    """Compute the offset label for the given device.

    Args:
        device_type (Optional[str]): Type of device. If None or 'EEG', returns default prefix.
        prefix (str): Prefix for the offset label. Defaults to 'starting_offset'.

    Returns:
        str: Offset label for the device.
    """
    if not device_type or device_type == 'EEG':
        return prefix
    return f"{prefix}_{device_type}"


def offset_device(label: str, prefix: str = 'starting_offset') -> str:
    """Given a label, determine the device type.

    Args:
        label (str): Label to parse.
        prefix (str): Expected prefix of the label. Defaults to 'starting_offset'.

    Returns:
        str: Device type extracted from the label.

    Raises:
        AssertionError: If label does not start with the given prefix.
    """
    assert label.startswith(
        prefix), "Label must start with the given prefix"
    try:
        idx = label.index('_', len(prefix))
        return label[idx + 1:]
    except ValueError:
        return 'EEG'


def starting_offsets_by_device(
    triggers: List[Trigger],
    device_types: Optional[List[str]] = None
) -> Dict[str, Trigger]:
    """Returns a dict of starting_offset triggers keyed by device type.

    Args:
        triggers (List[Trigger]): List of triggers to search through.
        device_types (Optional[List[str]]): List of device types to include in the result.
            If provided, an entry is created for each one, using a default offset of 0.0
            if a match is not found.

    Returns:
        Dict[str, Trigger]: Dictionary mapping device types to their offset triggers.
    """
    offset_triggers = {}
    for trg in triggers:
        if trg.type == TriggerType.OFFSET:
            device_type = offset_device(trg.label)
            offset_triggers[device_type] = trg

    if device_types:
        data = {}
        for device_type in device_types:
            data[device_type] = offset_triggers.get(
                device_type,
                Trigger(offset_label(device_type), TriggerType.OFFSET, 0.0))
        return data
    return offset_triggers


def find_starting_offset(
    triggers: List[Trigger],
    device_type: Optional[str] = None
) -> Trigger:
    """Given a list of raw trigger data, determine the starting offset for the
    given device. The returned trigger has the timestamp of the first sample
    recorded for the device.

    Args:
        triggers (List[Trigger]): List of trigger data; should include Triggers of
            TriggerType.OFFSET.
        device_type (Optional[str]): Each device will generally have a different offset. This
            parameter is used to determine which trigger to use. If not given
            the EEG offset will be used by default. Ex. 'EYETRACKER'.

    Returns:
        Trigger: The Trigger for the first matching offset for the given device, or a
            Trigger with offset of 0.0 if a matching offset was not found.
    """
    label = offset_label(device_type)
    for trg in triggers:
        if trg.type == TriggerType.OFFSET and trg.label == label:
            return trg
    log.info(f"Offset not found (device_type: {device_type}); using 0.0")
    return Trigger(label, TriggerType.OFFSET, 0.0)


def read(path: str) -> List[Trigger]:
    """Read all Triggers from the given text file.

    Args:
        path (str): Trigger (.txt) file to read.

    Returns:
        List[Trigger]: List of triggers read from the file.

    Raises:
        FileNotFoundError: If the file does not exist or is not a .txt file.
    """
    if not path.endswith('.txt') or not os.path.exists(path):
        raise FileNotFoundError(
            f'Valid triggers .txt file not found at [{path}].')
    with open(path, encoding=DEFAULT_ENCODING) as raw_txt:
        triggers = read_data(raw_txt)
    return triggers


def apply_offsets(
    triggers: List[Trigger],
    starting_offset: Trigger,
    static_offset: float = 0.0
) -> List[Trigger]:
    """Returns a list of triggers with timestamps adjusted relative to the
    device start time. Offset triggers are filtered out if present.

    Args:
        triggers (List[Trigger]): List of triggers to adjust.
        starting_offset (Trigger): Offset from the device start time.
        static_offset (float): The measured static system offset.

    Returns:
        List[Trigger]: List of triggers with timestamps relative to the starting_offset.
    """
    total_offset = starting_offset.time + static_offset
    return [
        trg.with_offset(total_offset) for trg in triggers
        if trg.type != TriggerType.OFFSET
    ]


def exclude_types(
    triggers: List[Trigger],
    types: Optional[List[TriggerType]] = None
) -> List[Trigger]:
    """Filter the list of triggers to exclude the provided types.

    Args:
        triggers (List[Trigger]): List of triggers to filter.
        types (Optional[List[TriggerType]]): List of trigger types to exclude.

    Returns:
        List[Trigger]: Filtered list of triggers.
    """
    if not types:
        return triggers
    return [trg for trg in triggers if trg.type not in types]


class TriggerHandler:
    """Class that contains methods to work with Triggers, including adding and
    writing triggers and loading triggers from a txt file.

    Attributes:
        encoding (str): File encoding to use.
        path (str): Path to the trigger file.
        file_name (str): Name of the trigger file.
        flush (FlushFrequency): Frequency at which to flush triggers to file.
        triggers (List[Trigger]): List of triggers being handled.
        file_path (str): Full path to the trigger file.
        file (TextIO): File handle for the trigger file.
    """

    encoding = DEFAULT_ENCODING

    def __init__(
        self,
        path: str,
        file_name: str,
        flush: FlushFrequency
    ) -> None:
        """Initialize the TriggerHandler.

        Args:
            path (str): Path to the trigger file.
            file_name (str): Name of the trigger file.
            flush (FlushFrequency): Frequency at which to flush triggers to file.

        Raises:
            Exception: If the file already exists.
        """
        self.path = path
        self.file_name = f'{file_name}.txt' if not file_name.endswith(
            '.txt') else file_name
        self.flush = flush
        self.triggers: List[Trigger] = []
        self.file_path = f'{self.path}/{self.file_name}'
        self.flush = flush

        if os.path.exists(self.file_name):
            raise Exception(f"[{self.file_name}] already exists, any writing "
                            "will overwrite data in the existing file.")

        self.file = open(self.file_path, 'w+', encoding=self.encoding)

    def close(self) -> None:
        """Close the trigger file and ensure all data is written."""
        self.write()
        self.file.close()

    def write(self) -> None:
        """Writes current Triggers in self.triggers[] to .txt file in self.file_name.
        File writes in the format "label, targetness, time".
        """
        for trigger in self.triggers:
            self.file.write(
                f'{trigger.label} {trigger.type.value} {trigger.time}\n')

        self.triggers = []

    @staticmethod
    def read_text_file(path: str) -> Tuple[List[Trigger], float]:
        """Read Triggers from the given text file.

        Args:
            path (str): Trigger (.txt) file to read.

        Returns:
            Tuple[List[Trigger], float]: List of triggers and offset time.
        """
        triggers = read(path)
        offset = find_starting_offset(triggers)
        triggers = exclude_types(triggers, [TriggerType.SYSTEM])
        return triggers, offset.time

    @staticmethod
    def load(
        path: str,
        offset: float = 0.0,
        exclusion: Optional[List[TriggerType]] = None,
        device_type: Optional[str] = None
    ) -> List[Trigger]:
        """Loads a list of triggers from a .txt of triggers.

        Args:
            path (str): Name or file path of .txt trigger file to be loaded.
                Input string must include file extension (.txt).
            offset (float): If desired, time offset for all loaded triggers,
                positive number for adding time, negative number for subtracting time.
            exclusion (Optional[List[TriggerType]]): If desired, list of TriggerType's
                to be removed from the loaded trigger list.
            device_type (Optional[str]): If specified looks for the starting_offset for
                a given device; default is to use the EEG offset.

        Returns:
            List[Trigger]: List of Triggers from loaded .txt file with desired modifications.
        """
        excluded_types = exclusion or []
        triggers = read(path)
        starting_offset = find_starting_offset(triggers, device_type)
        return apply_offsets(exclude_types(triggers, excluded_types),
                             starting_offset,
                             static_offset=offset)

    def add_triggers(self, triggers: List[Trigger]) -> List[Trigger]:
        """Adds provided list of Triggers to self.triggers.

        Args:
            triggers (List[Trigger]): List of Trigger objects to be added to the
                handler's list of Triggers (self.triggers).

        Returns:
            List[Trigger]: Returns list of Triggers currently part of Handler.
        """
        self.triggers.extend(triggers)

        if self.flush is FlushFrequency.EVERY:
            self.write()

        return self.triggers


def convert_timing_triggers(
    timing: List[Tuple[str, float]],
    target_stimuli: str,
    trigger_type: Callable
) -> List[Trigger]:
    """Convert Stimuli Times to Triggers.

    Using the stimuli presentation times provided by the display, convert them into BciPy Triggers.

    Args:
        timing (List[Tuple[str, float]]): List of (symbol, time) tuples.
        target_stimuli (str): Target stimulus identifier.
        trigger_type (Callable): Function to determine trigger type.

    Returns:
        List[Trigger]: List of converted triggers.
    """
    return [
        Trigger(symbol, trigger_type(symbol, target_stimuli, i), time)
        for i, (symbol, time) in enumerate(timing)
    ]


def load_triggers(
    trigger_path: str,
    remove_pre_fixation: bool = True,
    offset: float = 0.0,
    exclusion: Optional[List[TriggerType]] = None,
    device_type: Optional[str] = None,
    apply_starting_offset: bool = True
) -> List[Trigger]:
    """Trigger Decoder.

    Given a path to trigger data, this method loads valid Triggers.

    Args:
        trigger_path (str): Path to triggers file.
        remove_pre_fixation (bool): Boolean to determine whether any stimuli before a fixation + system should be removed.
        offset (float): Static offset value to apply to triggers.
        exclusion (Optional[List[TriggerType]]): Any TriggerTypes to be filtered from data returned.
        device_type (Optional[str]): Used to determine which starting_offset value to use; if
            a 'starting_offset' trigger is found it will be applied.
        apply_starting_offset (bool): If False, does not apply the starting offset for
            the given device_type.

    Returns:
        List[Trigger]: List of processed triggers.
    """
    excluded_types = exclusion or []
    excluded_types += TriggerType.pre_fixation() if remove_pre_fixation else [
        TriggerType.SYSTEM
    ]

    triggers = read(trigger_path)
    starting_offset = Trigger('', TriggerType.OFFSET, 0.0)
    if apply_starting_offset:
        starting_offset = find_starting_offset(triggers, device_type)

    filtered = exclude_types(triggers, excluded_types)
    corrected = apply_offsets(filtered, starting_offset, static_offset=offset)
    return corrected


def trigger_decoder(
    trigger_path: str,
    remove_pre_fixation: bool = True,
    offset: float = 0.0,
    exclusion: Optional[List[TriggerType]] = None,
    device_type: Optional[str] = None,
    apply_starting_offset: bool = True
) -> Tuple[List[str], List[float], List[str]]:
    """Trigger Decoder.

    Given a path to trigger data, this method loads valid Triggers and returns their type, timing and label.

    Args:
        trigger_path (str): Path to triggers file.
        remove_pre_fixation (bool): Boolean to determine whether any stimuli before a fixation + system should be removed.
        offset (float): Static offset value to apply to triggers.
        exclusion (Optional[List[TriggerType]]): Any TriggerTypes to be filtered from data returned.
        device_type (Optional[str]): Used to determine which starting_offset value to use; if
            a 'starting_offset' trigger is found it will be applied.
        apply_starting_offset (bool): If False, does not apply the starting offset for
            the given device_type.

    Returns:
        Tuple[List[str], List[float], List[str]]: Tuple containing trigger types, timings, and labels.
    """
    triggers = load_triggers(trigger_path, remove_pre_fixation, offset,
                             exclusion, device_type, apply_starting_offset)

    labels, types, times = zip(*triggers)
    return list(map(str, types)), list(times), list(labels)
