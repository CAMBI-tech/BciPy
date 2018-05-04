import os
from psychopy import visual, event

import numpy as np


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
        if parameters['is_txt_sti']['value'] == 'false':
            # construct an array of paths to images
            path = parameters['path_to_presentation_images']['value']
            image_array = []
            for image_filename in os.listdir(path):
                if image_filename.endswith(".png"):
                    image_array.append(os.path.join(path, image_filename))

            return image_array

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
    time2 = (last_stim_time + 2) * daq.device_info.fs

    # Construct triggers to send off for processing
    triggers = [(text, timing - time1)
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
                raise Exception("Not enough data recieved")

        # Take only the sensor data from raw data and transpose it
        raw_data = np.array([np.array(raw_data[i][0]) for i in
                             range(len(raw_data))]).transpose()

    except Exception as e:
        print("Error in daq: get_data()")
        raise e

    return raw_data, triggers, target_info


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
        height=float(parameters['txt_height']['value']),
        text=parameters['trial_complete_message']['value'],
        font=parameters['font_text']['value'],
        pos=(float(parameters['pos_text_x']['value']),
             float(parameters['pos_text_y']['value'])),
        wrapWidth=None,
        color=parameters['trial_complete_message_color']['value'],
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
