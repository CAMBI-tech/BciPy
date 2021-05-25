import os
from typing import Any, List, Union, Tuple, Optional
import logging
import random
import numpy as np
from psychopy import core, event, visual
from string import ascii_uppercase

from bcipy.signal.model import InputDataType
from bcipy.tasks.exceptions import InsufficientDataException
from itertools import zip_longest

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
        what that frequency should be based in the presentation
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
    if parameters:
        if not parameters['is_txt_stim']:
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


def process_data_for_decision(
        inquiry_timing,
        daq,
        window,
        parameters,
        first_session_stim_time,
        static_offset=None,
        buf_length=None):
    """Process Data for Decision.

    Processes the raw data (triggers and EEG) into a form that can be passed to
    signal processing and classifiers.

    Parameters
    ----------
        inquiry_timing(array): array of tuples containing stimulus timing and
            text
        daq (object): data acquisition object
        window: window to reactivate if deactivated by windows
        parameters: parameters dictionary
        first_session_stim_time (float): time that the first stimuli was presented
            for the session. Used to calculate offsets.
        static_offset (float): offset present in the system which should be accounted for when
            creating data for classification
        buf_length: length of data needed after the last sample in order to reshape correctly

    Returns
    -------
        (raw_data, triggers, target_info) tuple
    """

    # Get timing of the first and last stimuli
    _, first_stim_time = inquiry_timing[0]
    _, last_stim_time = inquiry_timing[-1]

    static_offset = static_offset or parameters['static_trigger_offset']
    # if there is an argument supplied for buffer length use that
    if buf_length:
        buffer_length = buf_length
    else:
        buffer_length = parameters['trial_length']

    # get any offset calculated from the daq
    daq_offset = daq.offset

    if daq_offset:
        offset = daq_offset - first_session_stim_time + static_offset
        time1 = (first_stim_time + offset) * daq.device_info.fs
        time2 = (last_stim_time + offset + buffer_length) * daq.device_info.fs
    else:
        time1 = (first_stim_time + static_offset) * daq.device_info.fs
        time2 = (last_stim_time + static_offset +
                 buffer_length) * daq.device_info.fs

    # Construct triggers to send off for processing
    triggers = [(text, ((timing) - first_stim_time))
                for text, timing in inquiry_timing]

    # Assign labels for triggers
    # TODO: This doesn't seem useful and is misleading
    target_info = ['nontarget'] * len(triggers)

    # Define the amount of data required for any processing to occur.
    data_limit = (last_stim_time - first_stim_time +
                  buffer_length) * daq.device_info.fs

    # Query for raw data
    try:
        # Call get_data method on daq with start/end
        raw_data = daq.get_data(start=time1, end=time2)

        # If not enough raw_data returned in the first query, let's try again
        #  using only the start param. This is known issue on Windows.
        #  #windowsbug
        if len(raw_data) < data_limit:

            # Call get_data method on daq with just start
            raw_data = daq.get_data(start=time1)

            # If there is still insufficient data returned, throw an error
            if len(raw_data) < data_limit:
                message = f'Process Data Error: Not enough data received to process. ' \
                          f'Data Limit = {data_limit}. Data received = {len(raw_data)}'
                log.error(message)
                raise InsufficientDataException(message)

        # Take only the sensor data from raw data and transpose it
        raw_data = np.array([np.array([_float_val(col) for col in record.data])
                             for record in raw_data],
                            dtype=np.float64).transpose()

    except Exception as e:
        log.error(f'Uncaught Error in Process Data for Decision: {e}')
        raise e

    return raw_data, triggers, target_info


def _float_val(col: Any) -> float:
    """Convert marker data to float values so we can put them in a
    typed np.array. The marker column has type float if it has a 0.0
    value, and would only have type str for a marker value."""
    if isinstance(col, str):
        return 1.0
    return float(col)


def trial_complete_message(win, parameters):
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
        height=parameters['task_height'],
        text=parameters['trial_complete_message'],
        font=parameters['task_font'],
        pos=(float(parameters['text_pos_x']),
             float(parameters['text_pos_y'])),
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
        clock: core.Clock,
        stamp_label: str = 'bcipy_key_press') -> Union[list, None]:
    """Get Key Press.

    A method to retrieve keys pressed of interest and get back a timestamp with
        a custom label


    Parameters
    ----------
        key_list(List[str]): list of keys to look for being pressed. Ex. ['space']
        clock(core.Clock): clock to use for timestamping any key press
        stamp_label(str): custom label to use for timstamping along with the key itself

    Returns
    -------
        Key Press Timing(List[stamp_label, timestamp])
    """
    response = event.getKeys(keyList=key_list, timeStamped=clock)
    if response:
        return [f'{stamp_label}_{response[0][0]}', response[0][1]]
    return None


def data_reshaper(input_data_type: InputDataType, *args, **kwargs):
    if input_data_type == InputDataType.TRIAL:
        return trial_reshaper(*args, **kwargs)
    elif input_data_type == InputDataType.INQUIRY:
        return inquiry_reshaper(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported input data type: {input_data_type}")


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def inquiry_reshaper(trial_target_info: list,
                     timing_info: list,
                     eeg_data: np.ndarray,
                     fs: int,
                     trials_per_inquiry: int,
                     offset: float = 0,
                     channel_map: List[int] = DEFAULT_CHANNEL_MAP,
                     trial_length: float = 0.5):
    """Extract inquiry data and labels.
    Args:
        trial_target_info (list): labels each trial as "target", "non-target", "first_pres_target", etc
        timing_info (list): Timestamp of each event in seconds
        eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
        fs (int): sample rate of preprocessed EEG data
        trials_per_inquiry (int): number of trials in each inquiry
        offset (float, optional): Any calculated or hypothesized offsets in timings. Defaults to 0.
        channel_map (tuple, optional): Describes which channels to include or discard. Defaults to DEFAULT_CHANNEL_MAP.
        trial_length (float, optional): [description]. Defaults to 0.5.

    Returns:
        reshaped_data (np.ndarray): inquiry data of shape (Channels, Inquiries, Samples)
        labels (np.ndarray): integer label for each inquiry. With trials_per_inquiry=K,
            a label of [0, K-1] indicates position of "target", or label of K indicates
            target was not present.
    """
    # Remove the channels that we are not interested in
    channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
    eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

    # Number of samples we are interested per inquiry
    num_samples = int(trial_length * fs * trials_per_inquiry)

    # Remove unwanted elements from target info and timing info
    targets_included = set(["target", "nontarget"])
    tmp_targets, tmp_timing = [], []
    for target, timing in zip(trial_target_info, timing_info):
        if target in targets_included:
            tmp_targets.append(target)
            tmp_timing.append(timing)
    trial_target_info = tmp_targets
    timing_info = tmp_timing

    n_inquiry = len(timing_info) // trials_per_inquiry

    # triggers in seconds are mapped to triggers in number of samples.
    triggers = list(map(lambda x: int((x + offset) * fs), timing_info))

    # shape (Channels, Inquiries, Samples)
    reshaped_data = np.zeros((len(eeg_data), n_inquiry, num_samples))

    # Label for every inquiry
    labels = np.zeros(n_inquiry)

    for inquiry_idx, chunk in enumerate(grouper(zip(trial_target_info, triggers), trials_per_inquiry)):
        # label is the index of the "target", or else the length of the inquiry
        label = trials_per_inquiry
        first_trigger = None
        for trial_idx, (target, trigger) in enumerate(chunk):
            if first_trigger is None:
                first_trigger = trigger

            if target == 'target':
                label = trial_idx

        labels[inquiry_idx] = label

        # For every channel append filtered channel data to trials
        reshaped_data[:, inquiry_idx, :] = eeg_data[:, first_trigger:first_trigger + num_samples]

    return reshaped_data, labels


def trial_reshaper(trial_target_info: list,
                   timing_info: list,
                   eeg_data: np.ndarray,
                   fs: int,
                   trials_per_inquiry: Optional[int] = None,
                   offset: float = 0,
                   channel_map: List[int] = DEFAULT_CHANNEL_MAP,
                   trial_length: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Extract trial data and labels.

    Args:
        trial_target_info (list): labels each trial as "target", "non-target", "first_pres_target", etc
        timing_info (list): Timestamp of each event in seconds
        eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
        fs (int): sample rate of preprocessed EEG data
        trials_per_inquiry (int, optional): unused, kept here for consistent interface with `inquiry_reshaper`
        offset (float, optional): Any calculated or hypothesized offsets in timings. Defaults to 0.
        channel_map (tuple, optional): Describes which channels to include or discard. Defaults to DEFAULT_CHANNEL_MAP.
        trial_length (float, optional): [description]. Defaults to 0.5.

    Returns:
        trial_data (np.ndarray): shape (channels, trials, samples) reshaped data
        labels (np.ndarray): integer label for each trial

    TODO ??? In all modes > calibration, we assume the timing info is given in samples not seconds.
    """
    # Remove the channels that we are not interested in
    channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
    eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

    # Number of samples we are interested per trial
    num_samples = int(trial_length * fs)

    # Remove unwanted elements from target info and timing info
    targets_included = set(["target", "nontarget"])
    tmp_targets, tmp_timing = [], []
    for target, timing in zip(trial_target_info, timing_info):
        if target in targets_included:
            tmp_targets.append(target)
            tmp_timing.append(timing)
    trial_target_info = tmp_targets
    timing_info = tmp_timing

    # triggers in seconds are mapped to triggers in number of samples.
    triggers = list(map(lambda x: int((x + offset) * fs), timing_info))

    # shape (Channels, Trials, Samples)
    reshaped_trials = np.zeros((len(eeg_data), len(triggers), num_samples))

    # Label for every trial
    labels = np.zeros(len(triggers))

    for trial_idx, (target, trigger) in enumerate(zip(trial_target_info, triggers)):
        # Assign targetness to labels for each trial
        if target == 'target':
            labels[trial_idx] = 1

        # For every channel append filtered channel data to trials
        reshaped_trials[:, trial_idx, :] = eeg_data[:, trigger:trigger + num_samples]

    return reshaped_trials, labels


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
