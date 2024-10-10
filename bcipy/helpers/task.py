import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from psychopy import core, event, visual

from bcipy.acquisition.multimodal import ClientManager, ContentType
from bcipy.acquisition.record import Record
from bcipy.config import MAX_PAUSE_SECONDS, SESSION_COMPLETE_MESSAGE
from bcipy.helpers.clock import Clock
from bcipy.helpers.stimuli import get_fixation
from bcipy.task.exceptions import InsufficientDataException

log = logging.getLogger(__name__)


def fake_copy_phrase_decision(copy_phrase: str, target_letter: str, text_task: str) -> Tuple[str, str, bool]:
    """Fake Copy Phrase Decision.

    Parameters
    ----------
        copy_phrase(str): phrase to be copied
        target_letter(str): the letter supposed to be typed
        text_task(str): phrase spelled at this time

    Returns
    -------
        (next_target_letter, text_task, run) tuple
    """
    if text_task == '*':
        length_of_spelled_letters = 0
    else:
        length_of_spelled_letters = len(text_task)

    length_of_phrase = len(copy_phrase)

    if length_of_spelled_letters == 0:
        text_task = copy_phrase[length_of_spelled_letters]
    else:
        text_task += copy_phrase[length_of_spelled_letters]

    length_of_spelled_letters += 1

    # If there is still text to be spelled, update the text_task
    # and target letter
    if length_of_spelled_letters < length_of_phrase:
        next_target_letter = copy_phrase[length_of_spelled_letters]

        run = True

    # else, end the run
    else:
        run = False
        next_target_letter = ''
        text_task = copy_phrase

    return next_target_letter, text_task, run


def calculate_stimulation_freq(flash_time: float) -> float:
    """Calculate Stimulation Frequency.

    In an RSVP paradigm, the inquiry itself will produce an
        SSVEP response to the stimulation. Here we calculate
        what that frequency should be based on the presentation
        time.

    PARAMETERS
    ----------
    :param: flash_time: time in seconds to present RSVP inquiry letters
    :returns: frequency: stimulation frequency of the inquiry
    """

    # We want to know how many stimuli will present in a second
    return 1 / flash_time


def construct_triggers(inquiry_timing: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Construct triggers from inquiry_timing data.

    Parameters
    ----------
    - inquiry_timing: list of tuples containing stimulus timing and text

    Returns
    -------
    list of (stim, offset) tuples, where offset is calculated relative to the
    first stim time.
    """
    if inquiry_timing:
        _, first_stim_time = inquiry_timing[0]
        return [(stim, ((timing) - first_stim_time))
                for stim, timing in inquiry_timing]
    return []


def target_info(triggers: List[Tuple[str, float]],
                target_letter: Optional[str] = None,
                is_txt: bool = True) -> List[str]:
    """Targetness for each item in triggers.

    Parameters
    ----------
    - triggers : list of (stim, offset)
    - target_letter : letter the user is attempting to spell
    - is_txt : bool indicating whether the triggers are text stimuli

    Returns
    -------
    list of ('target' | 'nontarget') for each trigger.
    """
    fixation = get_fixation(is_txt)
    labels = {target_letter: 'target', fixation: 'fixation'}
    return [labels.get(trg[0], 'nontarget') for trg in triggers]


def get_data_for_decision(inquiry_timing: List[Tuple[str, float]],
                          daq: ClientManager,
                          offset: float = 0.0,
                          prestim: float = 0.0,
                          poststim: float = 0.0) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Queries the acquisition client for a slice of data and processes the
    resulting raw data into a form that can be passed to signal processing and
    classifiers.

    Parameters
    ----------
    - inquiry_timing(list): list of tuples containing stimuli timing and labels. We assume the list progresses in
    - daq (DataAcquisitionClient): bcipy data acquisition client
    - offset (float): offset present in the system which should be accounted for when creating data for classification.
        This is determined experimentally.
    - prestim (float): length of data needed before the first sample to reshape and apply transformations
    - poststim (float): length of data needed after the last sample in order to reshape and apply transformations

    Returns
    -------
    (raw_data, triggers) tuple
    """
    _, first_stim_time = inquiry_timing[0]
    _, last_stim_time = inquiry_timing[-1]

    # adjust for offsets
    time1 = first_stim_time + offset - prestim
    time2 = last_stim_time + offset

    if time2 < time1:
        raise InsufficientDataException(
            f'Invalid data query [{time1}-{time2}] with parameters:'
            f'[inquiry={inquiry_timing}, offset={offset}, prestim={prestim}, poststim={poststim}]')

    # Construct triggers to send off for processing. This should not be zero anymore. it would be for prestim_len = 0
    triggers = [(text, ((timing - first_stim_time) + prestim))
                for text, timing in inquiry_timing]

    # Define the amount of data required for any processing to occur.
    data_limit = round((time2 - time1 + poststim) * daq.device_spec.sample_rate)
    log.info(f'Need {data_limit} records for processing')

    # Query for raw data
    raw_data = daq.get_data(start=time1, limit=data_limit)

    if len(raw_data) < data_limit:
        message = f'Process Data Error: Not enough data received to process. ' \
            f'Data Limit = {data_limit}. Data received = {len(raw_data)}'
        log.error(message)
        raise InsufficientDataException(message)

    # Take only the sensor data from raw data and transpose it
    raw_data = np.array([
        np.array([_float_val(col) for col in record.data])
        for record in raw_data
    ],
        dtype=np.float64).transpose()

    return raw_data, triggers


def get_device_data_for_decision(
        inquiry_timing: List[Tuple[str, float]],
        daq: ClientManager,
        prestim: float = 0.0,
        poststim: float = 0.0) -> Dict[ContentType, List[Record]]:
    """Queries the acquisition client manager for a slice of data from each
    device and processes the resulting raw data into a form that can be passed
    to signal processing and classifiers.

    Parameters
    ----------
    - inquiry_timing(list): list of tuples containing stimuli timing and labels.
    - daq (ClientManager): bcipy data acquisition client manager
    - offset (float): offset present in the system which should be accounted for when creating data for classification.
        This is determined experimentally.
    - prestim (float): length of data needed before the first sample to reshape and apply transformations
    - poststim (float): length of data needed after the last sample in order to reshape and apply transformations

    Returns
    -------
        a dict mapping device content_type to the data;
            data has shape C x L where C is number of channels and L is the
            signal length.
    """
    _, first_stim_time = inquiry_timing[0]
    _, last_stim_time = inquiry_timing[-1]

    # adjust for offsets
    time1 = first_stim_time - prestim
    time2 = last_stim_time

    if time2 < time1:
        raise InsufficientDataException(
            f'Invalid data query [{time1}-{time2}] with parameters:'
            f'[inquiry={inquiry_timing}, prestim={prestim}, poststim={poststim}]'
        )

    data = daq.get_data_by_device(start=time1,
                                  seconds=(time2 - time1 + poststim),
                                  strict=True)
    for content_type in data.keys():
        # Take only the sensor data from raw data and transpose it.
        # TODO: Does this apply to all content types? If not this step could be
        # moved to the evidence evaluators.
        data[content_type] = np.array([
            np.array([_float_val(col) for col in record.data])
            for record in data[content_type]
        ],
            dtype=np.float64).transpose()

    return data


def relative_triggers(inquiry_timing: List[Tuple[str, float]],
                      prestim: float) -> List[Tuple[str, float]]:
    """Adjust the provided inquiry_timing triggers for processing. The new
    timing values are relative to the first stim time, rather than using
    absolute clock times.

    Parameters
    ----------
        inquiry_timing - list of (symbol, timestamp) pairs for each trigger in
            an inquiry
        prestim - seconds of data needed before the first sample to reshape and
            apply transformations
    """
    _, first_stim_time = inquiry_timing[0]
    return [(symbol, (timestamp - first_stim_time) + prestim)
            for symbol, timestamp in inquiry_timing]


def _float_val(col: Any) -> float:
    """Convert marker data to float values so we can put them in a
    typed np.array. The marker column has type float if it has a 0.0
    value, and would only have type str for a marker value."""
    if isinstance(col, str):
        return 1.0
    return float(col)


def trial_complete_message(win, parameters) -> List[visual.TextStim]:
    """Trial Complete Message.

    Function return a TextStim Object (see Psychopy) to complete the trial.

    Parameters
    ----------

        win (object): Psychopy Window Object, should be the same as the one
            used in the experiment
        parameters (dict): Dictionary of session parameters

    Returns
    -------
        array of message_stim (trial complete message to be displayed).
    """
    message_stim = visual.TextStim(
        win=win,
        height=parameters['info_height'],
        text=SESSION_COMPLETE_MESSAGE,
        font=parameters['font'],
        pos=(parameters['info_pos_x'],
             parameters['info_pos_y']),
        wrapWidth=None,
        color=parameters['info_color'],
        colorSpace='rgb',
        opacity=1, depth=-6.0)
    return [message_stim]


def print_message(window: visual.Window, message: str = "Initializing..."):
    """Draws a message on the display window using default config.

    Parameters
    ----------
        window (object): Psychopy Window Object, should be the same as the one
            used in the experiment
        parameters (dict): Dictionary of session parameters

    Returns
    -------
        TextStim object
    """
    message_stim = visual.TextStim(win=window, text=message)
    message_stim.draw()
    window.flip()
    return message_stim


def get_user_input(window, message, color, first_run=False):
    """Get User Input.

    Function returns whether or not to stop a trial. If a key of interest is
        passed (e.g. space), it will act on it.

    Parameters
    ----------
        window[psychopy task window]: task window.  *assumes wait_screen method
        message: message to display in the wait screen
        color: wait screen color
        first_run: the first_run will always pause.

    Returns
    -------
        True to continue the task, False to stop and exit.
    """
    if first_run:
        return pause_on_wait_screen(window, message, color)
    else:
        keys = event.getKeys(keyList=['space', 'escape'])
        if keys:
            if keys[0] == 'space':
                return pause_on_wait_screen(window, message, color)
            if keys[0] == 'escape':
                return False
    return True


def pause_on_wait_screen(window, message, color) -> bool:
    """Pause on the wait screen until the user presses the Space key to resume,
    or the Escape key to exit.

    Returns
    -------
        True to resume; False to exit.
    """
    pause_start = time.time()
    window.wait_screen(message, color)
    keys = event.waitKeys(keyList=['space', 'escape'])

    elapsed_seconds = time.time() - pause_start
    if elapsed_seconds >= MAX_PAUSE_SECONDS:
        log.info(f"Pause exceeded the allowed time ({MAX_PAUSE_SECONDS} seconds). Ending task.")
        return False
    if keys[0] == 'escape':
        return False

    return True


def get_key_press(
        key_list: List[str],
        clock: Clock,
        stamp_label: str = 'bcipy_key_press') -> Union[list, None]:
    """Get Key Press.

    A method to retrieve keys pressed of interest and get back a timestamp with
        a custom label


    Parameters
    ----------
        key_list(List[str]): list of keys to look for being pressed. Ex. ['space']
        clock(Clock): clock to use for timestamping any key press
        stamp_label(str): custom label to use for timstamping along with the key itself

    Returns
    -------
        Key Press Timing(List[stamp_label, timestamp])
    """
    response = event.getKeys(keyList=key_list, timeStamped=True)
    if response:
        # The timestamp from the response uses the psychopy.core.monotonicClock
        # which records the number of seconds since the experiment start (core
        # was imported).
        key, stamp = response[0]
        offset = clock.getTime() - core.getTime()
        timestamp = stamp + offset
        return [f'{stamp_label}_{key}', timestamp]
    return None


def pause_calibration(window, display, current_index: int, parameters: dict):
    """Pause calibration.

    Pauses calibration for a given number of seconds and displays a countdown
    to the user.


    PARAMETERS
    ----------
    :param: window: Currently active PsychoPy window
    :param: display: The current display
    :param: current_index: number of trials that have already taken place
    :param: trials_before_break: number of trials before break
    :param: break_len: length of the break time (in seconds)
    :param: break_message: message to display to the user during the break

    :returns: bool: break has taken place
    """
    # Check whether or not to present a break
    trials_before_break = parameters['trials_before_break']
    break_len = parameters['break_len']
    break_message = parameters['break_message']

    if (current_index != 0) and (current_index % trials_before_break) == 0:

        # present break message for break length
        for counter in range(break_len):
            time = break_len - counter
            message = f'{break_message} {time}s'
            display.update_task_state(
                text=message,
                color_list=['white'])
            display.draw_static()
            window.flip()
            core.wait(1)
        return True

    return False


def generate_targets(alp, stim_number):
    """Generate a list of targets for each trial, minimizing duplication."""
    if stim_number <= len(alp):
        return random.sample(alp, stim_number)

    # minimize duplicates
    times, remainder = divmod(stim_number, len(alp))

    lists = [random.sample(alp, len(alp)) for _ in range(times)]
    lists.append(random.sample(alp, remainder))

    # flatten list of lists
    targets = [target for sublist in lists for target in sublist]

    return targets


def consecutive_incorrect(target_text: str, spelled_text: str) -> int:
    """Function that computes the number of consecutive symbols that
    are incorrectly spelled.

    >>> consecutive_incorrect('WORLD', 'H')
    1
    >>> consecutive_incorrect('WORLD', 'W')
    0
    >>> consecutive_incorrect('WORLD', 'WOHL')
    2
    """
    if not target_text:
        return len(spelled_text)
    for i, character in enumerate(spelled_text):
        if character != target_text[i]:
            return len(spelled_text[i:])
    return 0
