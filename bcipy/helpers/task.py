import os
from typing import Any
import logging

import numpy as np
from psychopy import core, event, visual

from bcipy.tasks.exceptions import InsufficientDataException

log = logging.getLogger(__name__)

SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'
DEFAULT_CHANNEL_MAP = (1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                       1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0)


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

    In an RSVP paradigm, the sequence itself will produce an
        SSVEP response to the stimulation. Here we calculate
        what that frequency should be based in the presentation
        time.

    PARAMETERS
    ----------
    :param: flash_time: time in seconds to present RSVP sequence letters
    :returns: frequency: stimulation frequency of the sequence
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

    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            BACKSPACE_CHAR, SPACE_CHAR]


def process_data_for_decision(
        sequence_timing,
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
        sequence_timing(array): array of tuples containing stimulus timing and
            text
        daq (object): data acquisition object
        window: window to reactivate if deactivated by windows
        parameters: parameters dictionary
        first_session_stim_time (float): time that the first stimuli was presented
            for the session. Used to calculate offsets.

    Returns
    -------
        (raw_data, triggers, target_info) tuple
    """

    # Get timing of the first and last stimuli
    _, first_stim_time = sequence_timing[0]
    _, last_stim_time = sequence_timing[-1]

    static_offset = static_offset or parameters['static_trigger_offset']
    # if there is an argument supplied for buffer length use that
    if buf_length:
        buffer_length = buf_length
    else:
        buffer_length = parameters['len_data_sequence_buffer']

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
                for text, timing in sequence_timing]

    # Assign labels for triggers
    # TODO: This doesn't seem useful and is misleading
    target_info = ['nontarget'] * len(triggers)

    # Define the amount of data required for any processing to occur.
    data_limit = (last_stim_time - first_stim_time +
                  buffer_length) * daq.device_info.fs

    # Query for raw data
    try:
        # Call get_data method on daq with start/end
        raw_data = daq.get_data(start=time1, end=time2, win=window)

        # If not enough raw_data returned in the first query, let's try again
        #  using only the start param. This is known issue on Windows.
        #  #windowsbug
        if len(raw_data) < data_limit:

            # Call get_data method on daq with just start
            raw_data = daq.get_data(start=time1, win=window)

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

    Function return a TextStim Object (see Psycopy) to complete the trial.

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


def trial_reshaper(trial_target_info: list,
                   timing_info: list,
                   eeg_data: np.array,
                   fs: int, k: int, mode: str,
                   offset: float = 0,
                   channel_map: tuple = DEFAULT_CHANNEL_MAP,
                   trial_length: float = 0.5) -> tuple:
    """Trial Reshaper.

    Trail reshaper is used to reshape trials, based on trial target info (target, non-target),
        timing information (sec), eeg_data, sampling rate (fs), down-sampling rate (k),
        mode of operation (ex. 'calibration'),
        offset (any calculated or hypothesized offsets in timings),
        channel map (which channels to include in reshaping) and
        trial length (length of reshaped trials in secs)

    In all modes > calibration, we assume the timing info is given in samples not seconds.

    :return (reshaped_trials, labels, num_of_sequences, trials_per_seq)

        reshaped_trials is a 3 dimensional np array where the first dimension
            is the channel, second dimension the trial/letter, and third
            dimension is time samples. So at 300hz and a presentation time of
            0.2 seconds per letter, the third dimension would have about 60
            samples.
            If this were a json structure it might look like:
                {
                    'ch1': {
                        'B' : [-647.123, -93.124, ...],
                        'T' : [...],
                        ...
                    },
                    'ch2': {
                        'B' : [39.60, -224.124, ...],
                        'T' : [...],
                        ...
                    },
                    ...
                }
    """
    try:
        # Remove the channels that we are not interested in
        channel_indexes_to_remove = []
        for channel_index in range(len(eeg_data)):
            if channel_map[channel_index] == 0:
                channel_indexes_to_remove.append(channel_index)

        eeg_data = np.delete(eeg_data,
                             channel_indexes_to_remove,
                             axis=0)
        after_filter_frequency = fs / k
        # Number of samples we are interested per trial
        num_samples = int(trial_length * after_filter_frequency)

        # MODE CALIBRATION
        if mode == 'calibration':
            trials_per_seq = []
            count = 0
            # Count every sequences trials
            for symbol_info in trial_target_info:
                if symbol_info == 'first_pres_target':
                    trials_per_seq.append(count)
                    count = 0
                elif symbol_info == 'nontarget' or 'target':
                    count += 1
                else:
                    raise Exception('Incorrectly formatted trigger file. \
                    See trial_reshaper documentation for expected input.')

            # The first element is garbage. Get rid of it.
            trials_per_seq = trials_per_seq[1:]
            # Append the last sequences trial number
            trials_per_seq.append(count)
            # Make the list a numpy array.
            trials_per_seq = np.array(trials_per_seq)

            # Mark every element in timing_info if 'first_pres_target'
            for symbol_info_index in range(len(trial_target_info)):
                if trial_target_info[symbol_info_index] == 'first_pres_target':
                    timing_info[symbol_info_index] = -1

            # Get rid of 'first_pres_target' trials information
            trial_target_info = list(filter(lambda x: x != 'first_pres_target',
                                            trial_target_info))
            timing_info = list(filter(lambda x: x != -1, timing_info))

            # triggers in seconds are mapped to triggers in number of samples.
            triggers = list(
                map(lambda x: int((x + offset) * after_filter_frequency), timing_info))

            # 3 dimensional np array first dimension is channels
            # second dimension is trials and third dimension is time samples.
            reshaped_trials = np.zeros(
                (len(eeg_data), len(triggers), num_samples))

            # Label for every trial
            labels = np.zeros(len(triggers))

            for trial in range(len(triggers)):
                # Assign targetness to labels for each trial
                if trial_target_info[trial] == 'target':
                    labels[trial] = 1

                # For every channel append filtered channel data to trials
                for channel in range(len(eeg_data)):
                    reshaped_trials[channel][trial] = \
                        eeg_data[channel][triggers[trial]:triggers[trial] + num_samples]

            num_of_sequences = int(sum(labels))

        # MODE COPY PHRASE
        elif mode == 'copy_phrase':

            # triggers in samples are mapped to triggers in number of filtered
            # samples.
            triggers = list(
                map(lambda x: int((x + offset) * after_filter_frequency), timing_info))

            # 3 dimensional np array first dimension is channels
            # second dimension is trials and third dimension is time samples.
            reshaped_trials = np.zeros(
                (len(eeg_data), len(triggers), num_samples))

            # Label for every trial
            labels = np.zeros(len(triggers))

            for trial in range(len(triggers)):
                # Assign targetness to labels for each trial
                if trial_target_info[trial] == 'target':
                    labels[trial] = 1

                # For every channel append filtered channel data to trials
                for channel in range(len(eeg_data)):
                    reshaped_trials[channel][trial] = \
                        eeg_data[channel][
                        triggers[trial]:triggers[trial] + num_samples]

            # In copy phrase, num of sequence is assumed to be 1.
            num_of_sequences = 1
            # Since there is only one sequence, all trials are in the sequence
            trials_per_seq = len(triggers)

        # MODE FREE SPELL
        elif mode == 'free_spell':

            # triggers in sample are mapped to triggers in number of filtered
            # samples.
            triggers = list(
                map(lambda x: int((x + offset) * after_filter_frequency), timing_info))

            # 3 dimensional np array first dimension is channels
            # second dimension is trials and third dimension is time samples.
            reshaped_trials = np.zeros(
                (len(eeg_data), len(triggers), num_samples))

            labels = None

            for trial in range(len(triggers)):

                # For every channel append filtered channel data to trials
                for channel in range(len(eeg_data)):
                    reshaped_trials[channel][trial] = \
                        eeg_data[channel][
                        triggers[trial]:triggers[trial] + num_samples]

            # In copy phrase, num of sequence is assumed to be 1.
            num_of_sequences = 1
            # Since there is only one sequence, all trials are in the sequence
            trials_per_seq = len(triggers)
        else:
            raise Exception(
                'Trial_reshaper does not work in this operating mode.')

        # Return our trials, labels and some useful information about the
        # arrays
        return reshaped_trials, labels, num_of_sequences, trials_per_seq

    except Exception as e:
        raise Exception(
            f'Could not reshape trial for mode: {mode}, {fs}, {k}. Error: {e}')


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
