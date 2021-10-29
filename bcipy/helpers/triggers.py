import logging
import os
from enum import Enum
from typing import Dict, List, Optional, TextIO, Tuple

from psychopy import core, visual

from bcipy.helpers.clock import Clock
from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.load import load_txt_data
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.raw_data import RawDataReader
from bcipy.helpers.stimuli import resize_image

log = logging.getLogger(__name__)

NONE_VALUES = ['0', '0.0']

CALIBRATION_IMAGE_PATH = 'bcipy/static/images/testing/white.png'
DEFAULT_CALIBRATION_TRIGGER_NAME = 'calibration_trigger'


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
                         trigger_name: str = DEFAULT_CALIBRATION_TRIGGER_NAME,
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


def _write_triggers_from_inquiry_calibration(array: list,
                                             trigger_file: TextIO,
                                             offset: bool = False):
    """Write triggers from calibration.

    Helper Function to write trigger data to provided trigger_file. It assigns
        target letter based on the first presented letter in inquiry, then
        assigns target/nontarget label to following letters.

    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp
    """

    x = 0

    if offset:
        # extract the letter and timing from the array
        (letter, time) = array
        targetness = 'offset_correction'
        trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")

    else:
        for i in array:

            # extract the letter and timing from the array
            (letter, time) = i

            # determine what the trigger are
            if letter == 'calibration_trigger':
                targetness = 'calib'
                target_letter = letter
            else:
                if x == 0:
                    targetness = 'first_pres_target'
                    target_letter = letter
                elif x == 1:
                    targetness = 'fixation'
                elif x > 1 and target_letter == letter:
                    targetness = 'target'
                else:
                    targetness = 'nontarget'

                x += 1

            # write to the trigger_file
            trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")

    return trigger_file


def _write_triggers_from_inquiry_copy_phrase(
        triggers: list,
        trigger_file: TextIO,
        copy_phrase: str,
        typed_text: str,
        offset: bool = False) -> TextIO:
    """Write triggers from copy phrase.

    Helper Function to write trigger data to provided trigger_file in a copy phrase task.
        It assigns target letter based on matching the next needed letter in typed text
        then assigns target/nontarget label to following letters. It's assumed if offset
        is true, only one record will be written (the offset_correction).

    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp
    """
    # write the offset and return the file
    if offset:
        # extract the letter and timing from the array
        (symbol, time) = triggers
        trigger_file.write(f'{symbol} offset_correction {time}\n')
        return trigger_file

    spelling_length = len(typed_text)
    last_typed = typed_text[-1] if typed_text else None
    target_symbol = copy_phrase[spelling_length - 1]

    # because there is the possibility of incorrect letters and correction,
    # we check here what is expected response.
    if last_typed == target_symbol or not last_typed:
        target_symbol = copy_phrase[spelling_length - 1]
    else:
        # If the last typed and target do not match and this is not the
        # symbols have been typed. The correct symbol is backspace.
        target_symbol = '<'

    for trigger in triggers:

        # extract the letter and timing from the array
        (symbol, time) = trigger

        # catch all the internal labels and assign targetness
        if symbol == 'calibration_trigger':
            targetness = 'calib'
        elif symbol == 'inquiry_preview':
            targetness = 'preview'
        # we write the key press + key so we check the prefix is in the symbol
        elif 'bcipy_key_press' in symbol:
            targetness = 'key_press'

        # assign targetness to the core symbols
        elif symbol == '+':
            targetness = 'fixation'
        elif target_symbol == symbol:
            targetness = 'target'
        else:
            targetness = 'nontarget'

        # write to the trigger_file
        trigger_file.write(f'{symbol} {targetness} {time}\n')

    return trigger_file


def _write_triggers_from_inquiry_free_spell(array, trigger_file):
    """
    Write triggers from free spell.

    Helper Function to write trigger data to provided trigger_file.

    It writes in the following order:
        (I) presented letter, (II) timestamp
    """

    for i in array:

        # extract the letter and timing from the array
        (letter, time) = i

        # write to trigger_file
        trigger_file.write('%s %s' % (letter, time) + "\n")

    return trigger_file


def write_triggers_from_inquiry_icon_to_icon(inquiry_timing: List[Tuple],
                                             trigger_file: TextIO,
                                             target: str,
                                             target_displayed: bool,
                                             offset=None):
    """
    Write triggers from icon to icon task.
    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp

    Parameters:
    ----------
        inquiry_timing - list of (icon, time) output from rsvp after
            displaying a inquiry.
        trigger_file - open file in which to write.
        target - target for the current inquiry
        target_displayed - whether or not the target was presented during the
            inquiry.
    """
    if offset:
        (letter, time) = inquiry_timing
        targetness = 'offset_correction'
        trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")
        return

    icons, _times = zip(*inquiry_timing)
    calib_presented = 'calibration_trigger' in icons
    calib_index = 0 if calib_presented else -1

    if calib_presented:
        target_pres_index = 1
        fixation_index = 2
    elif target_displayed:
        target_pres_index = 0
        fixation_index = 1
    else:
        target_pres_index = -1
        fixation_index = 0

    for i, (icon, presentation_time) in enumerate(inquiry_timing):
        targetness = 'nontarget'
        if i == calib_index:
            targetness = 'calib'
        elif i == target_pres_index:
            targetness = 'first_pres_target'
        elif i == fixation_index:
            targetness = 'fixation'
        elif icon == target:
            targetness = 'target'
        else:
            targetness = 'nontarget'
        trigger_file.write('%s %s %s' % (icon, targetness, presentation_time) +
                           "\n")


def trigger_decoder(mode: str, trigger_path: str = None, remove_pre_fixation=True) -> tuple:
    """Trigger Decoder.

    Given a mode of operation (calibration, copy phrase, etc) and
        a path to the trigger location (*.txt file), this function
        will split into symbols (A, ..., Z), timing info (32.222), and
        targetness (target, nontarget). It will also extract any saved
        offset information and pass that back.

    PARAMETERS
    ----------
    :param: mode: mode of bci operation. Note the mode changes how triggers
        are saved.
    :param: trigger_path: [Optional] path to triggers.txt file
    :return: tuple: symbol_info, trial_target_info, timing_info, offset.
    """

    # Load triggers.txt
    if not trigger_path:
        trigger_path = load_txt_data()

    # Get every line of trigger.txt
    with open(trigger_path, 'r+') as text_file:
        # most trigger files has three columns:
        #   SYMBOL, TARGETNESS_INFO[OPTIONAL], TIMING
        trigger_txt = [line.split() for line in text_file]

    # extract stimuli from the text.
    if remove_pre_fixation:
        stimuli_triggers = [
            line for line in trigger_txt
            if line[1] == 'target' or line[1] == 'nontarget'
        ]
    else:
        stimuli_triggers = [
            line for line in trigger_txt
            if line[0] != 'calibration_trigger' and line[0] != 'offset'
        ]

    # from the stimuli array, pull our the symbol information
    symbol_info = list(map(lambda x: x[0], stimuli_triggers))

    # If operating mode is free spell, it only has 2 columns
    #   otherwise, it has 3
    if mode != 'free_spell':
        trial_target_info = list(map(lambda x: x[1], stimuli_triggers))
        timing_info = list(map(lambda x: eval(x[2]), stimuli_triggers))
    else:
        trial_target_info = None
        timing_info = list(map(lambda x: eval(x[1]), stimuli_triggers))

    # Get any offset or calibration triggers
    offset_array = [line[2] for line in trigger_txt if line[0] == 'offset']
    calib_trigger_array = [
        line[2] for line in trigger_txt if line[0] == 'calibration_trigger'
    ]

    # If present, calculate the offset between the DAQ and Triggers from
    # display
    if len(offset_array) == 1 and len(calib_trigger_array) == 1:

        # Extract the offset and calibration trigger time
        offset_time = float(offset_array[0])
        calib_trigger_time = float(calib_trigger_array[0])

        # Calculate the offset (ASSUMES DAQ STARTED FIRST!)
        offset = offset_time - calib_trigger_time

    # Otherwise, assume no observed offset
    else:
        offset = 0

    return symbol_info, trial_target_info, timing_info, offset


def apply_trigger_offset(timing: List[float], offset: float):
    """Apply trigger offset.

    Due to refresh rates and clock differences between display and acquisition, offsets in the system exist.
    This method takes a list of trigger times and adds the offset to them.
    """
    corrected_timing = []
    for time in timing:
        corrected_timing.append(time + offset)
    return corrected_timing


class Labeller:
    """Labels the targetness for a trigger value in a raw_data file."""

    def __init__(self):
        super(Labeller, self).__init__()

    def label(self, trigger):
        raise NotImplementedError('Subclass must define the label method')


class LslCalibrationLabeller(Labeller):
    """Calculates targetness for calibration data. Uses a state machine to
    determine how to label triggers.

    Parameters:
    -----------
        inq_len: stim_length parameter value for the experiment; used to calculate
            targetness for first_pres_target.
    """

    def __init__(self, inq_len: int):
        super(LslCalibrationLabeller, self).__init__()
        self.inq_len = inq_len
        self.prev = None
        self.current_target = None
        self.inq_position = 0

    def label(self, trigger):
        """Calculates the targetness for the given trigger, accounting for the
        previous triggers/states encountered."""
        state = ''
        if self.prev is None:
            # First trigger is always calibration.
            state = 'calib'
        elif self.prev == 'calib':
            self.current_target = trigger
            state = 'first_pres_target'
        elif self.prev == 'first_pres_target':
            # reset the inquiry when fixation '+' is encountered.
            self.inq_pos = 0
            state = 'fixation'
        else:
            self.inq_pos += 1
            if self.inq_pos > self.inq_len:
                self.current_target = trigger
                state = 'first_pres_target'
            elif trigger == self.current_target:
                state = 'target'
            else:
                state = 'nontarget'
        self.prev = state
        return state


class LslCopyPhraseLabeller(Labeller):
    """Sequentially calculates targetness for copy phrase triggers."""

    def __init__(self, copy_text: str, typed_text: str):
        super(LslCopyPhraseLabeller, self).__init__()
        self.copy_text = copy_text
        self.typed_text = typed_text
        self.prev = None

        self.pos = 0
        self.typing_pos = -1  # inquiry length should be >= typed text.

        self.current_target = None

    def label(self, trigger):
        """Calculates the targetness for the given trigger, accounting for the
        previous triggers/states encountered."""
        state = ''
        if self.prev is None:
            state = 'calib'
        elif trigger == '+':
            self.typing_pos += 1
            if not self.current_target:
                # set target to first letter in the copy phrase.
                self.current_target = self.copy_text[self.pos]
            else:
                last_typed = self.typed_text[self.typing_pos - 1]
                if last_typed == self.current_target:
                    # increment if the user typed the target correctly
                    if last_typed != '<':
                        self.pos += 1
                    self.current_target = self.copy_text[self.pos]
                else:
                    # Error correction.
                    self.current_target = '<'

            state = 'fixation'
        else:
            if trigger == self.current_target:
                state = 'target'
            else:
                state = 'nontarget'
        self.prev = state
        return state


def _extract_triggers(csvfile: str,
                      trg_field,
                      labeller: Labeller) -> List[Tuple[str, str, str]]:
    """Extracts trigger data from an experiment output csv file.
    Parameters:
    -----------
        csvfile: path to raw data file
        trg_field: optional; name of the data column with the trigger data;
                   defaults to 'TRG'
        labeller: Labeller used to calculate the targetness value for a
            given trigger.
    Returns:
    --------
        list of tuples of (trigger, targetness, timestamp)
    """
    data = []

    with RawDataReader(csvfile) as reader:
        trg_index = reader.columns.index(trg_field)
        timestamp_index = reader.columns.index('timestamp')

        for row in reader:
            trg = row[trg_index]
            if trg not in NONE_VALUES:
                if 'calibration' in trg:
                    trg = 'calibration_trigger'
                targetness = labeller.label(trg)
                data.append((trg, targetness, row[timestamp_index]))

    return data


def write_trigger_file_from_lsl_calibration(csvfile: str,
                                            trigger_file: TextIO,
                                            inq_len: int,
                                            trg_field: str = 'TRG'):
    """Creates a triggers.txt file from TRG data recorded in the raw_data
    output from a calibration."""
    extracted = extract_from_calibration(csvfile, inq_len, trg_field)
    _write_trigger_file_from_extraction(trigger_file, extracted)


def write_trigger_file_from_lsl_copy_phrase(csvfile: str,
                                            trigger_file: TextIO,
                                            copy_text: str,
                                            typed_text: str,
                                            trg_field: str = 'TRG'):
    """Creates a triggers.txt file from TRG data recorded in the raw_data
    output from a copy phrase."""
    extracted = extract_from_copy_phrase(csvfile, copy_text, typed_text,
                                         trg_field)
    _write_trigger_file_from_extraction(trigger_file, extracted)


def _write_trigger_file_from_extraction(
        trigger_file: TextIO, extraction: List[Tuple[str, str, str]]):
    """Writes triggers that have been extracted from a raw_data file to a
    file."""
    for trigger, targetness, timestamp in extraction:
        trigger_file.write(f"{trigger} {targetness} {timestamp}\n")

    # TODO: is this assumption correct?
    trigger_file.write("offset offset_correction 0.0")


def extract_from_calibration(csvfile: TextIO,
                             inq_len: int,
                             trg_field: str = 'TRG'
                             ) -> List[Tuple[str, str, str]]:
    """Extracts trigger data from a calibration output csv file.
    Parameters:
    -----------
        csvfile: path to the raw data file.
        inq_len: stim_length parameter value for the experiment; used to calculate
                 targetness for first_pres_target.
        trg_field: optional; name of the data column with the trigger data;
                   defaults to 'TRG'
    Returns:
    --------
        list of tuples of (trigger, targetness, timestamp), where timestamp is
        the timestamp recorded in the file.
    """

    return _extract_triggers(csvfile,
                             trg_field,
                             labeller=LslCalibrationLabeller(inq_len))


def extract_from_copy_phrase(csvfile: str,
                             copy_text: str,
                             typed_text: str,
                             trg_field: str = 'TRG'
                             ) -> List[Tuple[str, str, str]]:
    """Extracts trigger data from a copy phrase output csv file.
    Parameters:
    -----------
        csvfile: path to raw data file.
        copy_text: phrase to copy
        typed_text: participant typed response
        trg_field: optional; name of the data column with the trigger data;
                   defaults to 'TRG'
    Returns:
    --------
        list of tuples of (trigger, targetness, timestamp), where timestamp is
        the timestamp recorded in the file.
    """
    labeller = LslCopyPhraseLabeller(copy_text, typed_text)
    return _extract_triggers(csvfile, trg_field, labeller=labeller)


def trigger_durations(params: Parameters) -> Dict[str, float]:
    """Duration for each type of trigger given in seconds."""
    return {
        'calib': 0.0,
        'first_pres_target': params['time_target'],
        'fixation': params['time_cross'],
        'nontarget': params['time_flash'],
        'target': params['time_flash']
    }


def read_triggers(triggers_file: TextIO) -> List[Tuple[str, str, float]]:
    """Read in the triggers.txt file. Convert the timestamps to be in
    acquisition clock units using the offset listed in the file (last entry).

    triggers_file - open triggers.txt

    Returns
    -------
        list of (symbol, targetness, stamp) tuples.
    """

    records = [line.split(' ') for line in triggers_file.readlines()]
    # calibration
    (_cname, _ctype, calibration_stamp) = records[0]
    (_acq_name, _acq_type, acq_stamp) = records.pop()
    offset = float(acq_stamp) - float(calibration_stamp)

    corrected = []
    for _, (name, trg_type, stamp) in enumerate(records):
        corrected.append((name, trg_type, float(stamp) + offset))
    return corrected




class TriggerType(Enum):
    """
    Enum for the primary types of Triggers.
    """

    NONTARGET = "nontarget"
    TARGET = "target"
    FIXATION = "fixation"
    PROMPT = "first_pres_target"

    @classmethod
    def list(cls):
        """Returns all enum values as a list"""
        return list(map(lambda c: c.value, cls))

    def __str__(self):
        return f'{self.value}'


class Trigger:
    """
    Object that encompasses data for a single trigger instance.
    """

    def __init__(self,
                 label: str,
                 type: TriggerType,
                 time: str):
        self.label = label
        self.type = type
        self.time = time

    def __repr__(self):
        return f'Trigger: label=[{self.label}] type=[{self.type}] time=[{self.time}]'


class FlushSensitivity(Enum):
    """
    Enum that defines how often list of Triggers will be written and dumped.
    """

    EVERY = "flush after every trigger addition"
    END = "flush at end of session"


class TriggerHandler:

    triggers = []

    def __init__(self,
                 path: str,
                 file_name: str,
                 flush_sens: FlushSensitivity):
        self.path = path
        self.file_name = f'{file_name}.txt'
        self.flush_sens = flush_sens
        self.file = None

    def __enter__(self):
        # Throws an error if file already exists
        if os.path.exists(self.file_name):
            raise Exception(f"[{self.file_name}] already exists, any writing "
                            "will overwrite data in the existing file.")

        self.file = open(self.file_name, 'w+')
        return self

    def __exit__(self, type, value, traceback):
        self.write()
        self.file.close()

    def write(self):
        """
        Writes current Triggers in self.triggers[] to .txt file in self.file_name.
        File writes in the format "label, targetness, time".

        Parameters
        ----------
        None

        Returns
        -------
            None
        """

        for trigger in self.triggers:
            self.file.write(f'{trigger.label} {trigger.type.value} {trigger.time}\n')

        # Something about this isn't working, trigger list isn't being cleared
        self.triggers = []

    def load(self,
             path: str,
             offset: Optional[float]=None,
             exclusion: Optional[List[TriggerType]]=None) -> List[Trigger]:
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
        exclusion (Optional List[TriggerType]): if desired, list of TriggerTypes
            to be removed from the loaded trigger list.

        Returns
        -------
            List of Triggers from loaded .txt file with desired modifications
        """

        # Checking for file with given path, with or without .txt
        if not path.endswith('.txt') or not os.path.exists(path):
            raise FileNotFoundError(f"Valid triggers .txt file not found at [{path}]."
                                    "\nPlease rerun program.")

        txt_list = []
        with open(path_checked) as raw_txt:
            for line in raw_txt:
                line_split = line.split()
                txt_list.append(line_split)

        if offset:
            # If there is exclusion but no offset,
            # Program would read exclusion as offset and throw error
            # This moves "offset" into its proper parameter of exclusion
            if isinstance(offset, list) and exclusion is None:
                exclusion = offset
            else:
                for item in txt_list:
                    time_float = float(item[2]) + offset
                    item[2] = str(time_float)

        if exclusion:
            for type in exclusion:
                for item in txt_list:
                    txt_list[:] = [item for item in txt_list if not type.value == item[1]]

        new_trigger_list = []
        for e in txt_list:
            new_trigger_list.append(Trigger(e[0],
                                    TriggerType(e[1]),
                                    str(e[2])))

        return new_trigger_list


    def add_triggers(self, triggers: List[Trigger]) -> List[Trigger]:
        """
        Adds given list of Triggers to self.triggers[]

        Parameters
        ----------
        triggers (List[Triggers]): list of Trigger objects to be added to the
            handler's list of Triggers, self.triggers[]

        Returns
        -------
            None
        """

        self.triggers.extend(triggers)

        if self.flush_sens is FlushSensitivity.EVERY:
            self.write()
