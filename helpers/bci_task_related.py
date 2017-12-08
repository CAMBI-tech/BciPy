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


def alphabet():
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']


def _process_data_for_decision(sequence_timing, daq):
    """Process Data for Decision.

    Processes the raw data (triggers and eeg) into a form that can be passed to
        signal processing and classifiers.

    Parameters
    ----------
        sequence_timing(array): array of tuples containing stimulus timing and
            text
        daq (object): data acquistion object

    Returns
    -------
        (raw_data, triggers, target_info) tuple
    """

    # Get timing of the first and last stimuli
    _, first_stim_time = sequence_timing[0]
    _, last_stim_time = sequence_timing[len(sequence_timing) - 1]

    # define my first and last time points #changeforrelease
    time1 = first_stim_time
    time2 = last_stim_time + 2

    # Construct triggers to send off for processing
    triggers = [(text, timing - time1)
                for text, timing in sequence_timing]

    # Assign labels for triggers
    target_info = ['nontarget'] * len(triggers)

    # Define the amount of data required for any processing to occur.
    data_limit = (last_stim_time - first_stim_time + .5) * daq._device.fs

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
        print "Error in daq: get_data()"
        raise e

    return raw_data, triggers, target_info
