import logging
import os
import random

from abc import ABC, abstractmethod
from itertools import zip_longest
from string import ascii_uppercase
from turtle import position
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
from psychopy import core, event, visual

from bcipy.helpers.clock import Clock
from bcipy.task.exceptions import InsufficientDataException

log = logging.getLogger(__name__)

SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'
DEFAULT_CHANNEL_MAP = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                       1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]


def fake_copy_phrase_decision(copy_phrase, target_letter, text_task):
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
        next_target_letter = None
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


def alphabet(parameters=None, include_path=True):
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    if parameters and not parameters['is_txt_stim']:
        # construct an array of paths to images
        path = parameters['path_to_presentation_images']
        stimulus_array = []
        for stimulus_filename in sorted(os.listdir(path)):
            # PLUS.png is reserved for the fixation symbol
            if stimulus_filename.endswith(
                    '.png') and not stimulus_filename.endswith('PLUS.png'):
                if include_path:
                    img = os.path.join(path, stimulus_filename)
                else:
                    img = os.path.splitext(stimulus_filename)[0]
                stimulus_array.append(img)
        return stimulus_array

    return list(ascii_uppercase) + [BACKSPACE_CHAR, SPACE_CHAR]


def construct_triggers(inquiry_timing: List[List]) -> List[Tuple[str, float]]:
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
                target_letter: str = None) -> List[str]:
    """Targetness for each item in triggers.

    Parameters
    ----------
    - triggers : list of (stim, offset)
    - target_letter : letter the user is attempting to spell

    Returns
    -------
    list of ('target' | 'nontarget') for each trigger.
    """
    if target_letter:
        return [
            'target' if trg[0] == target_letter else 'nontarget'
            for trg in triggers
        ]
    return ['nontarget'] * len(triggers)


def get_data_for_decision(inquiry_timing,
                          daq,
                          offset=0.0,
                          prestim=0.0,
                          poststim=0.0):
    """Queries the acquisition client for a slice of data and processes the
    resulting raw data into a form that can be passed to signal processing and
    classifiers.

    Parameters
    ----------
    - inquiry_timing(list): list of tuples containing stimuli timing and labels. We assume the list progresses in
    - daq (DataAcquisitionClient): bcipy data acquisition client with a get_data method and device_info with fs defined
    - offset (float): offset present in the system which should be accounted for when creating data for classification; this is determined experimentally.
    - prestim (float): amount of of data needed before the first sample to reshape correctly
    - poststim: length of data needed after the last sample in order to reshape correctly

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
    triggers = [(text, ((timing) - first_stim_time))
                for text, timing in inquiry_timing]

    # Define the amount of data required for any processing to occur.
    data_limit = round((time2 - time1 + poststim) * daq.device_info.fs)
    log.debug(f'Need {data_limit} records for processing')

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
        text=parameters['trial_complete_message'],
        font=parameters['info_font'],
        pos=(parameters['info_pos_x'],
             parameters['info_pos_y']),
        wrapWidth=None,
        color=parameters['trial_complete_message_color'],
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

    Returns
    -------
        True/False: whether or not to stop a trial (based on escape key).
    """
    if not first_run:
        pause = False
        # check user input to make sure we should be going
        keys = event.getKeys(keyList=['space', 'escape'])

        if keys:
            # pause?
            if keys[0] == 'space':
                pause = True

            # escape?
            if keys[0] == 'escape':
                return False

    else:
        pause = True

    while pause:
        window.wait_screen(message, color)
        keys = event.getKeys(keyList=['space', 'escape'])

        if keys:
            if keys[0] == 'escape':
                return False
            pause = False

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


class Reshaper(ABC):
    @abstractmethod
    def __call__(self,
                 trial_labels: List[str],
                 timing_info: List[float],
                 eeg_data: np.ndarray,
                 fs: int,
                 trials_per_inquiry: int,
                 offset: float = 0,
                 channel_map: List[int] = DEFAULT_CHANNEL_MAP,
                 trial_length: float = 0.5,
                 target_label: str = "target",
                 labels_included: Set[str] = set(["target", "nontarget"]),
                 labels_excluded: Set[str] = set([])) -> Tuple[np.ndarray, np.ndarray]:
        ...


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class InquiryReshaper(Reshaper):
    def __call__(self,
                 trial_labels: List[str],
                 timing_info: List[float],
                 eeg_data: np.ndarray,
                 fs: int,
                 trials_per_inquiry: int,
                 offset: float = 0,
                 channel_map: List[int] = DEFAULT_CHANNEL_MAP,
                 trial_length: float = 0.5,
                 target_label: str = "target",
                 labels_included: Set[str] = set(["target", "nontarget"]),
                 labels_excluded: Set[str] = set([])) -> Tuple[np.ndarray, np.ndarray]:
        """Extract inquiry data and labels.

        Args:
            trial_labels (List[str]): labels each trial as "target", "non-target", "first_pres_target", etc
            timing_info (List[float]): Timestamp of each event in seconds
            eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
            fs (int): sample rate of EEG data. If data is downsampled, the sample rate should be also be downsampled.
            trials_per_inquiry (int): number of trials in each inquiry
            offset (float, optional): Any calculated or hypothesized offsets in timings. Defaults to 0.
            channel_map (List[int], optional): Describes which channels to include or discard.
                Defaults to DEFAULT_CHANNEL_MAP.
            trial_length (float, optional): [description]. Defaults to 0.5.
            target_label (str): label of target symbol. Defaults to "target"
            labels_included (Set[str]): labels to include. Defaults to "target" and "nontarget"
            labels_excluded (Set[str]): labels to exclude. Defaults to empty set.

        Returns:
            reshaped_data (np.ndarray): inquiry data of shape (Channels, Inquiries, Samples)
            labels (np.ndarray): integer label for each inquiry. With `trials_per_inquiry=K`,
                a label of [0, K-1] indicates the position of `target_label`, or label of K indicates
                `target_label` was not present.
        """
        # Remove the channels that we are not interested in
        channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
        eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

        # Remove unwanted elements from target info and timing info
        tmp_labels, tmp_timing = [], []
        for label, timing in zip(trial_labels, timing_info):
            if label in labels_included and label not in labels_excluded:
                tmp_labels.append(label)
                tmp_timing.append(timing)
        trial_labels = tmp_labels
        timing_info = tmp_timing

        n_inquiry = len(timing_info) // trials_per_inquiry
        trial_duration_samples = int(trial_length * fs)

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = list(map(lambda x: int((x + offset) * fs), timing_info))

        # Label for every inquiry
        labels = np.zeros(n_inquiry, dtype=np.long)
        reshaped_data = []
        for inquiry_idx, trials_within_inquiry in enumerate(grouper(zip(trial_labels, triggers), trials_per_inquiry)):
            # label is the index of the "target", or else the length of the inquiry
            inquiry_label = trials_per_inquiry
            for trial_idx, (trial_label, trigger) in enumerate(trials_within_inquiry):
                if trial_label == target_label:
                    inquiry_label = trial_idx

            # Inquiry lasts from first trial onset until final trial onset + trial_length
            first_trigger = trials_within_inquiry[0][1]
            last_trigger = trials_within_inquiry[-1][1]
            labels[inquiry_idx] = inquiry_label

            reshaped_data.append(eeg_data[:, first_trigger: last_trigger + trial_duration_samples])

        return np.stack(reshaped_data, 1), labels


class TrialReshaper(Reshaper):
    def __call__(self,
                 trial_labels: list,
                 timing_info: list,
                 eeg_data: np.ndarray,
                 fs: int,
                 trials_per_inquiry: Optional[int] = None,
                 offset: float = 0,
                 channel_map: List[int] = DEFAULT_CHANNEL_MAP,
                 trial_length: float = 0.5,
                 target_label: str = "target",
                 labels_included: Set[str] = set(["target", "nontarget"]),
                 labels_excluded: Set[str] = set([])) -> Tuple[np.ndarray, np.ndarray]:
        """Extract trial data and labels.

        Args:
            trial_labels (list): labels each trial as "target", "non-target", "first_pres_target", etc
            timing_info (list): Timestamp of each event in seconds
            eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
            fs (int): sample rate of preprocessed EEG data
            trials_per_inquiry (int, optional): unused, kept here for consistent interface with `inquiry_reshaper`
            offset (float, optional): Any calculated or hypothesized offsets in timings.
                Defaults to 0.
            channel_map (tuple, optional): Describes which channels to include or discard.
                Defaults to DEFAULT_CHANNEL_MAP.
            trial_length (float, optional): [description]. Defaults to 0.5.
            target_label (str): label of target symbol. Defaults to "target"
            labels_included (Set[str]): labels to include. Defaults to "target" and "nontarget"
            labels_excluded (Set[str]): labels to exclude. Defaults to empty set.

        Returns:
            trial_data (np.ndarray): shape (channels, trials, samples) reshaped data
            labels (np.ndarray): integer label for each trial
        """
        # Remove the channels that we are not interested in
        channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
        eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

        # Number of samples we are interested per trial
        num_samples = int(trial_length * fs)

        # Remove unwanted elements from target info and timing info
        tmp_labels, tmp_timing = [], []
        for label, timing in zip(trial_labels, timing_info):
            if label in labels_included and label not in labels_excluded:
                tmp_labels.append(label)
                tmp_timing.append(timing)
        trial_labels = tmp_labels
        timing_info = tmp_timing

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = list(map(lambda x: int((x + offset) * fs), timing_info))

        # Label for every trial
        labels = np.zeros(len(triggers), dtype=np.long)
        reshaped_trials = []
        for trial_idx, (trial_label, trigger) in enumerate(zip(trial_labels, triggers)):
            if trial_label == target_label:
                labels[trial_idx] = 1

            # For every channel append filtered channel data to trials
            reshaped_trials.append(eeg_data[:, trigger: trigger + num_samples])

        return np.stack(reshaped_trials, 1), labels


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
    if (stim_number <= len(alp)):
        return random.sample(alp, stim_number)

    # minimize duplicates
    times, remainder = divmod(stim_number, len(alp))

    lists = [random.sample(alp, len(alp)) for _ in range(times)]
    lists.append(random.sample(alp, remainder))

    # flatten list of lists
    targets = [target for sublist in lists for target in sublist]

    return targets
