import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

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

    def __str__(self) -> str:
        return f'{self.value}'


@dataclass(frozen=True)
class Trigger:
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
        if not path.endswith('.txt') or not os.path.exists(path):
            raise FileNotFoundError(f'Valid triggers .txt file not found at [{path}].')

        with open(path, encoding=TriggerHandler.encoding) as raw_txt:
            triggers = []
            for i, line in enumerate(raw_txt):
                try:
                    trg = Trigger.from_list(line.split())
                    triggers.append(trg)
                except (AssertionError, ValueError) as trg_error:
                    raise BciPyCoreException(
                        f'Error reading trigger on line {i+1} of {path}: {trg_error}') from trg_error

        # find next offset values in list and return or create a trigger with offset of 0.0
        offset = next(
            filter(lambda trg: trg.type == TriggerType.OFFSET, triggers),
            Trigger('starting_offset', TriggerType.OFFSET, 0.0))
        triggers = [trg for trg in triggers if trg.type != TriggerType.SYSTEM]

        return triggers, offset.time

    @staticmethod
    def load(path: str,
             offset: Optional[float] = 0.0,
             exclusion: Optional[List[TriggerType]] = None) -> List[Trigger]:
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

        Returns
        -------
            List of Triggers from loaded .txt file with desired modifications
        """

        # Checking for file with given path, with or without .txt
        triggers, system_offset = TriggerHandler.read_text_file(path)
        excluded_types = exclusion or []
        total_offset = offset + system_offset
        return [
            trg.with_offset(total_offset) for trg in triggers
            if trg.type not in excluded_types
        ]

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


def convert_timing_triggers(timing: List[tuple], target_stimuli: str, trigger_type: Callable) -> List[Trigger]:
    """Convert Stimuli Times to Triggers.

    Using the stimuli presentation times provided by the display, convert them into BciPy Triggers.
    """
    return [
        Trigger(symbol, trigger_type(symbol, target_stimuli, i), time) for i, (symbol, time) in enumerate(timing)
    ]


def trigger_decoder(trigger_path: str, remove_pre_fixation: bool = True, offset: float = 0.0,
                    exclusion: List[TriggerType] = []) -> Tuple[list, list, list]:
    """Trigger Decoder.

    Given a path to trigger data, this method loads valid Triggers and returns their type, timing and label.

    PARAMETERS
    ----------
    :param: trigger_path: path to triggers file
    :param: remove_pre_fixation: boolean to determine whether any stimuli before a fixation + system should be removed
    :param: offset: additional offset value to apply to triggers. If a valid 'starting_offset' present in the trigger
        file this will be applied be default.
    :param: exclusion [Optional]: any TriggerTypes to be filtered from data returned
    :return: tuple: trigger_type, trigger_timing, trigger_label
    """
    if remove_pre_fixation:
        exclusion += [TriggerType.FIXATION, TriggerType.PROMPT, TriggerType.SYSTEM, TriggerType.OFFSET]
    else:
        exclusion += [TriggerType.SYSTEM]
    triggers = TriggerHandler.load(trigger_path, offset=offset, exclusion=exclusion)

    # from the stimuli array, pull out the symbol information
    trigger_type = [trigger.type.value for trigger in triggers]
    trigger_label = [trigger.label for trigger in triggers]
    trigger_timing = [trigger.time for trigger in triggers]

    return trigger_type, trigger_timing, trigger_label
