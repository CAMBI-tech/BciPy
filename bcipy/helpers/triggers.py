import logging
import os
from enum import Enum
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from psychopy import core, visual

from bcipy.config import DEFAULT_ENCODING
from bcipy.helpers.clock import Clock
from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import resize_image

log = logging.getLogger(__name__)

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
    def list(cls):
        """Returns all enum values as a list"""
        return list(map(lambda c: c.value, cls))


class TriggerCallback:
    timing = None
    first_time = True

    def callback(self, clock: Clock, stimuli: str) -> None:
        if self.first_time:
            self.timing = [stimuli, clock.getTime()]
            self.first_time = False

    def reset(self):
        self.timing = None
        self.first_time = True


def _calibration_trigger(experiment_clock: Clock,
                         trigger_type: str = CalibrationType.TEXT.value,
                         trigger_name: str = 'calibration',
                         trigger_time: float = 1,
                         display=None,
                         on_trigger=None) -> List[tuple]:
    """Calibration Trigger.

    Outputs triggers for the purpose of calibrating data and stimuli.
    This is an ongoing difficulty between OS, DAQ devices and stimuli type. This
    code aims to operationalize the approach to finding the correct DAQ samples in
    relation to our trigger code.

    PARAMETERS
    ---------
    experiment_clock(clock): clock with getTime() method, which is used in the code
        to report timing of stimuli
    trigger_type(string): type of trigger that is desired (text, image, etc)
    trigger_name(string): name of the trigger used for callbacks / labeling
    trigger_time(float): time to display the trigger. Can also be used as a buffer.
    display(DisplayWindow): a window that can display stimuli. Currently, a Psychopy window.
    on_trigger(function): optional callback; if present gets called
                when the calibration trigger is fired; accepts a single
                parameter for the timing information.
    Return:
        timing(array): timing values for the calibration triggers to be written to trigger file or
                used to calculate offsets.
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
        calibration_box.size = resize_image(CALIBRATION_IMAGE_PATH, display.size, 0.75)

        display.callOnFlip(trigger_callback.callback, experiment_clock, trigger_name)
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
    return trigger_callback.timing


def trigger_durations(params: Parameters) -> Dict[str, float]:
    """Duration for each type of trigger given in seconds."""
    return {
        'offset': 0.0,
        'preview': params['preview_inquiry_length'],
        'fixation': params['time_fixation'],
        'prompt': params['time_prompt'],
        'nontarget': params['time_flash'],
        'target': params['time_flash']
    }


class TriggerType(Enum):
    """
    Enum for the primary types of Triggers.
    """

    NONTARGET = "nontarget"
    TARGET = "target"
    FIXATION = "fixation"
    PROMPT = "prompt"
    SYSTEM = "system"
    OFFSET = "offset"
    EVENT = "event"
    PREVIEW = "preview"

    @classmethod
    def list(cls) -> List[str]:
        """Returns all enum values as a list"""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def pre_fixation(cls) -> List['TriggerType']:
        """Returns the subset of TriggerTypes that occur before and including
        the FIXATION trigger."""
        return [
            TriggerType.FIXATION, TriggerType.PROMPT, TriggerType.SYSTEM,
            TriggerType.OFFSET
        ]

    def __str__(self) -> str:
        return f'{self.value}'


class Trigger(NamedTuple):
    """
    Object that encompasses data for a single trigger instance.
    """

    label: str
    type: TriggerType
    time: float

    def __repr__(self):
        return f'Trigger: label=[{self.label}] type=[{self.type}] time=[{self.time}]'

    def with_offset(self, offset: float):
        """Construct a copy of this Trigger with the offset adjusted."""
        return Trigger(self.label, self.type, self.time + offset)

    @classmethod
    def from_list(cls, lst: List[str]):
        """Constructs a Trigger from a serialized representation.

        Parameters
        ----------
            lst - serialized representation [label, type, stamp]
        """
        assert len(lst) == 3, "Input must have a label, type, and stamp"
        return cls(lst[0], TriggerType(lst[1]), float(lst[2]))


class FlushFrequency(Enum):
    """
    Enum that defines how often list of Triggers will be written and dumped.
    """

    EVERY = "flush after every trigger addition"
    END = "flush at end of session"


def read_data(lines: Iterable[str]) -> List[Trigger]:
    """Read raw trigger data from the given source.

    Parameters
    ----------
        data - iterable object where each item is a str with data for a single
            trigger.

    Returns
    -------
        list of all Triggers in the data.
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
    """
    if not device_type or device_type == 'EEG':
        return prefix
    return f"{prefix}_{device_type}"


def offset_device(label: str, prefix: str = 'starting_offset') -> str:
    """Given an label, determine the device type"""
    assert label.startswith(
        prefix), "Label must start with the given prefix"
    try:
        idx = label.index('_', len(prefix))
        return label[idx + 1:]
    except ValueError:
        return 'EEG'


def starting_offsets_by_device(
        triggers: List[Trigger],
        device_types: Optional[List[str]] = None) -> Dict[str, Trigger]:
    """Returns a dict of starting_offset triggers keyed by device type.

    If device_types are provided, an entry is created for each one, using a
    default offset of 0.0 if a match is not found.
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


def find_starting_offset(triggers: List[Trigger],
                         device_type: Optional[str] = None) -> Trigger:
    """Given a list of raw trigger data, determine the starting offset for the
    given device. The returned trigger has the timestamp of the first sample
    recorded for the device.

    If no device is provided the EEG offset will be used. If there are
    no offset triggers in the given data a Trigger with offset of 0.0 will be
    returned.

    Parameters
    ----------
        triggers - list of trigger data; should include Triggers of
          TriggerType.OFFSET
        device_type - each device will generally have a different offset. This
            parameter is used to determine which trigger to use. If not given
            the EEG offset will be used by default. Ex. 'EYETRACKER'
    Returns
    -------
        The Trigger for the first matching offset for the given device, or a
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

    Parameters
    ----------
        path - trigger (.txt) file to read
    Returns
    -------
        triggers
    """
    if not path.endswith('.txt') or not os.path.exists(path):
        raise FileNotFoundError(
            f'Valid triggers .txt file not found at [{path}].')
    with open(path, encoding=DEFAULT_ENCODING) as raw_txt:
        triggers = read_data(raw_txt)
    return triggers


def apply_offsets(triggers: List[Trigger],
                  starting_offset: Trigger,
                  static_offset: float = 0.0) -> List[Trigger]:
    """Returns a list of triggers with timestamps adjusted relative to the
    device start time. Offset triggers are filtered out if present.

    Parameters
    ----------
        triggers - list of triggers
        starting_offset - offset from the device start time.
        static_offset - the measured static system offset

    Returns
    -------
        a list of triggers with timestamps relative to the starting_offset
    """
    total_offset = starting_offset.time + static_offset
    return [
        trg.with_offset(total_offset) for trg in triggers
        if trg.type != TriggerType.OFFSET
    ]


def exclude_types(triggers: List[Trigger],
                  types: Optional[List[TriggerType]] = None) -> List[Trigger]:
    """Filter the list of triggers to exclude the provided types"""
    if not types:
        return triggers
    return [trg for trg in triggers if trg.type not in types]


class TriggerHandler:
    """
    Class that contains methods to work with Triggers, including adding and
    writing triggers and loading triggers from a txt file.
    """

    encoding = DEFAULT_ENCODING

    def __init__(self,
                 path: str,
                 file_name: str,
                 flush: FlushFrequency):
        self.path = path
        self.file_name = f'{file_name}.txt' if not file_name.endswith('.txt') else file_name
        self.flush = flush
        self.triggers = []
        self.file_path = f'{self.path}/{self.file_name}'
        self.flush = flush
        self.triggers = []

        if os.path.exists(self.file_name):
            raise Exception(f"[{self.file_name}] already exists, any writing "
                            "will overwrite data in the existing file.")

        self.file = open(self.file_path, 'w+', encoding=self.encoding)

    def close(self) -> None:
        """Close.

        Ensures all data is written and file is closed properly.
        """
        self.write()
        self.file.close()

    def write(self) -> None:
        """
        Writes current Triggers in self.triggers[] to .txt file in self.file_name.
        File writes in the format "label, targetness, time".
        """

        for trigger in self.triggers:
            self.file.write(f'{trigger.label} {trigger.type.value} {trigger.time}\n')

        self.triggers = []

    @staticmethod
    def read_text_file(path: str) -> Tuple[List[Trigger], float]:
        """Read Triggers from the given text file.
        Parameters
        ----------
            path - trigger (.txt) file to read
        Returns
        -------
            triggers, offset
        """
        triggers = read(path)
        offset = find_starting_offset(triggers)
        triggers = exclude_types(triggers, [TriggerType.SYSTEM])
        return triggers, offset.time

    @staticmethod
    def load(path: str,
             offset: Optional[float] = 0.0,
             exclusion: Optional[List[TriggerType]] = None,
             device_type: Optional[str] = None) -> List[Trigger]:
        """
        Loads a list of triggers from a .txt of triggers.

        Exclusion based on type only (ex. exclusion=[TriggerType.Fixation])

        1. Checks if .txt file exists at path
        2. Loads the triggers data as a list of lists
        3. If offset provided, adds it to the time as float
        4. If exclusion provided, filters those triggers
        5. Casts all loaded and modified triggers to Trigger
        6. Returns as a List[Triggers]

        Parameters
        ----------
        path (str): name or file path of .txt trigger file to be loaded.
            Input string must include file extension (.txt).
        offset (Optional float): if desired, time offset for all loaded triggers,
            positive number for adding time, negative number for subtracting time.
        exclusion (Optional List[TriggerType]): if desired, list of TriggerType's
            to be removed from the loaded trigger list.
        device_type : optional; if specified looks for the starting_offset for
            a given device; default is to use the EEG offset.

        Returns
        -------
            List of Triggers from loaded .txt file with desired modifications
        """
        excluded_types = exclusion or []
        triggers = read(path)
        starting_offset = find_starting_offset(triggers, device_type)
        return apply_offsets(exclude_types(triggers, excluded_types),
                             starting_offset,
                             static_offset=offset)

    def add_triggers(self, triggers: List[Trigger]) -> List[Trigger]:
        """
        Adds provided list of Triggers to self.triggers.

        Parameters
        ----------
        triggers (List[Triggers]): list of Trigger objects to be added to the
            handler's list of Triggers (self.triggers).

        Returns
        -------
            Returns list of Triggers currently part of Handler
        """
        self.triggers.extend(triggers)

        if self.flush is FlushFrequency.EVERY:
            self.write()

        return self.triggers


def convert_timing_triggers(timing: List[tuple], target_stimuli: str,
                            trigger_type: Callable) -> List[Trigger]:
    """Convert Stimuli Times to Triggers.

    Using the stimuli presentation times provided by the display, convert them into BciPy Triggers.
    """
    return [
        Trigger(symbol, trigger_type(symbol, target_stimuli, i), time)
        for i, (symbol, time) in enumerate(timing)
    ]


def trigger_decoder(
        trigger_path: str,
        remove_pre_fixation: bool = True,
        offset: float = 0.0,
        exclusion: Optional[List[TriggerType]] = None,
        device_type: Optional[str] = None,
        apply_starting_offset: bool = True) -> Tuple[list, list, list]:
    """Trigger Decoder.

    Given a path to trigger data, this method loads valid Triggers and returns their type, timing and label.

    Parameters
    ----------
        trigger_path: path to triggers file
        remove_pre_fixation: boolean to determine whether any stimuli before a fixation + system should be removed
        offset: static offset value to apply to triggers.
        exclusion: any TriggerTypes to be filtered from data returned
        device_type: used to determine which starting_offset value to use; if
            a 'starting_offset' trigger is found it will be applied.
        apply_starting_offset: if False, does not apply the starting offset for
            the given device_type.
    Returns
    -------
        tuple: trigger_type, trigger_timing, trigger_label
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

    labels, types, times = zip(*corrected)
    return list(map(str, types)), list(times), list(labels)
