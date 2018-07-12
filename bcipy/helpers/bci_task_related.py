import os
from psychopy import visual, event
import numpy as np
from typing import Any


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
    if text_task is '*':
        length_of_spelled_letters = 0
    else:
        length_of_spelled_letters = len(text_task)

    length_of_phrase = len(copy_phrase)

    if length_of_spelled_letters is 0:
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


def alphabet(parameters=None):
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    if parameters:
        if not parameters['is_txt_sti']:
            # construct an array of paths to images and wav files
            path = parameters['path_to_presentation_images']
            stimulus_array = []
            for stimulus_filename in os.listdir(path):
                if stimulus_filename.endswith(".png") or stimulus_filename.endswith(".wav"):
                    stimulus_array.append(os.path.join(path, stimulus_filename))

            return stimulus_array

    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_']


def process_data_for_decision(sequence_timing, daq):
    """Process Data for Decision.

    Processes the raw data (triggers and eeg) into a form that can be passed to
    signal processing and classifiers.

    Parameters
    ----------
        sequence_timing(array): array of tuples containing stimulus timing and
            text
        daq (object): data acquisition object

    Returns
    -------
        (raw_data, triggers, target_info) tuple
    """
    # Get timing of the first and last stimuli
    _, first_stim_time = sequence_timing[0]
    _, last_stim_time = sequence_timing[len(sequence_timing) - 1]

    # define my first and last time points #changeforrelease
    time1 = first_stim_time * daq.device_info.fs
    time2 = (last_stim_time + .5) * daq.device_info.fs

    # Construct triggers to send off for processing
    triggers = [(text, ((timing * daq.device_info.fs) - time1))
                for text, timing in sequence_timing]

    # Assign labels for triggers
    target_info = ['nontarget'] * len(triggers)

    # Define the amount of data required for any processing to occur.
    data_limit = (last_stim_time - first_stim_time + .5) * daq.device_info.fs

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
                raise Exception("Not enough data received")

        # Take only the sensor data from raw data and transpose it;
        raw_data = np.array([np.array([_float_val(col) for col in record.data])
                             for record in raw_data],
                            dtype=np.float64).transpose()

    except Exception as e:
        print("Error in daq: get_data()")
        raise e

    return raw_data, triggers, target_info


def _float_val(col: Any) -> float:
    """Convert marker data to float values so we can put them in a
    typed np.array. The marker column has type float if it has a 0.0
    value, and would only have type str for a marker value."""
    if isinstance(col, str):
        return 1.0
    else:
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
        height=float(parameters['txt_height']),
        text=parameters['trial_complete_message'],
        font=parameters['font_text'],
        pos=(float(parameters['pos_text_x']),
             float(parameters['pos_text_y'])),
        wrapWidth=None,
        color=parameters['trial_complete_message_color'],
        colorSpace='rgb',
        opacity=1, depth=-6.0)
    return [message_stim]

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
                   filtered_eeg: np.array,
                   fs: int, k: int, mode: str,
                   offset: float=0,
                   channel_map: tuple=(
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0), trial_length: float=0.5) -> tuple:
    """Trial Reshaper.

    Trail reshaper is used to reshape trials, based on trial target info (target, non-target),
        timing information (sec), filtered_eeg, sampling rate (fs), down-sampling rate (k),
        mode of operation (ex. 'calibration'),
        offset (any calculated or hypothesized offsets in timings),
        channel map (which channels to include in reshaping) and
        trial length (length of reshaped trials in secs)

    In all modes > calibration, we assume the timing info is given in samples not seconds.

    :return (reshaped_trials, labels, num_of_sequences, trials_per_seq)

    """
    try:
        # Remove the channels that we are not interested in
        channel_indexes_to_remove = []
        for channel_index in range(len(filtered_eeg)):
            if channel_map[channel_index] == 0:
                channel_indexes_to_remove.append(channel_index)

        # define our filtered eeg and frequency
        filtered_eeg = np.delete(filtered_eeg,
                                 channel_indexes_to_remove, axis=0)
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
                map(lambda x: int((x - offset) * after_filter_frequency), timing_info))

            # 3 dimensional np array first dimension is channels
            # second dimension is trials and third dimension is time samples.
            reshaped_trials = np.zeros(
                (len(filtered_eeg), len(triggers), num_samples))

            # Label for every trial
            labels = np.zeros(len(triggers))

            for trial in range(len(triggers)):
                # Assign targetness to labels for each trial
                if trial_target_info[trial] == 'target':
                    labels[trial] = 1

                # For every channel append filtered channel data to trials
                for channel in range(len(filtered_eeg)):
                    reshaped_trials[channel][trial] = \
                        filtered_eeg[channel][
                        triggers[trial]:triggers[trial] + num_samples]

            num_of_sequences = int(sum(labels))

        # MODE COPY PHRASE
        elif mode == 'copy_phrase':

            # triggers in samples are mapped to triggers in number of filtered samples.
            triggers = list(map(lambda x: int((x - offset) / k), timing_info))

            # 3 dimensional np array first dimension is channels
            # second dimension is trials and third dimension is time samples.
            reshaped_trials = np.zeros(
                (len(filtered_eeg), len(triggers), num_samples))

            # Label for every trial
            labels = np.zeros(len(triggers))

            for trial in range(len(triggers)):
                # Assign targetness to labels for each trial
                if trial_target_info[trial] == 'target':
                    labels[trial] = 1

                # For every channel append filtered channel data to trials
                for channel in range(len(filtered_eeg)):
                    reshaped_trials[channel][trial] = \
                        filtered_eeg[channel][
                        triggers[trial]:triggers[trial] + num_samples]

            # In copy phrase, num of sequence is assumed to be 1.
            num_of_sequences = 1
            # Since there is only one sequence, all trials are in the sequence
            trials_per_seq = len(triggers)

        # MODE FREE SPELL
        elif mode == 'free_spell':

            # triggers in sample are mapped to triggers in number of filtered samples.
            triggers = list(map(lambda x: int((x - offset) / k), timing_info))

            # 3 dimensional np array first dimension is channels
            # second dimension is trials and third dimension is time samples.
            reshaped_trials = np.zeros(
                (len(filtered_eeg), len(triggers), num_samples))

            labels = None

            for trial in range(len(triggers)):

                # For every channel append filtered channel data to trials
                for channel in range(len(filtered_eeg)):
                    reshaped_trials[channel][trial] = \
                        filtered_eeg[channel][
                        triggers[trial]:triggers[trial] + num_samples]

            # In copy phrase, num of sequence is assumed to be 1.
            num_of_sequences = 1
            # Since there is only one sequence, all trials are in the sequence
            trials_per_seq = len(triggers)
        else:
            raise Exception(
                'Trial_reshaper does not work in this operating mode.')

        # Return our trials, labels and some useful information about the arrays
        return reshaped_trials, labels, num_of_sequences, trials_per_seq

    except Exception as e:
        raise Exception(
            f'Could not reshape trial for mode: {mode}, {fs}, {k}. Error: {e}')
