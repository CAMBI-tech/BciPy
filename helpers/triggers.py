# -*- coding: utf-8 -*-
from helpers.load import load_txt_data


def _write_triggers_from_sequence_calibration(array, trigger_file):
    """
    Write triggers from calibration.

    Helper Function to write trigger data to provided trigger_file. It assigns
        target letter based on the first presented letter in sequence, then
        assigns target/nontarget label to following letters.

    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp
    """

    x = 0
    for i in array:

        # extract the letter and timing from the array
        (letter, time) = i

        # determine what the trigger are
        if x == 0:
            targetness = 'first_pres_target'
            target_letter = letter
        elif x == 1:
            targetness = 'fixation'
        elif x > 1 and target_letter == letter:
            targetness = 'target'
        else:
            targetness = 'nontarget'

        # write to the trigger_file
        trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")

        x += 1

    return trigger_file


def _write_triggers_from_sequence_copy_phrase(array, trigger_file,
                                              copy_text, typed_text):
    """
    Write triggers from copy phrase.

    Helper Function to write trigger data to provided trigger_file. It assigns
        target letter based on matching the next needed letter in typed text
        then assigns target/nontarget label to following letters.

    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp
    """

    # get relevant spelling info to determine what was and should be typed
    spelling_length = len(typed_text)
    last_typed = typed_text[-1]
    correct_letter = copy_text[spelling_length - 1]

    # because there is the possiblility of incorrect letter and correction,
    # we check here what is appropriate as a correct response
    if last_typed == correct_letter:
        correct_letter = copy_text[spelling_length]
    else:
        correct_letter = '<'

    x = 0

    for i in array:

        # extract the letter and timing from the array
        (letter, time) = i

        # determine what the triggers are:
        #       assumes there is no target letter presentation.
        if x == 0:
            targetness = 'fixation'
        elif x > 1 and correct_letter == letter:
            targetness = 'target'
        else:
            targetness = 'nontarget'

        # write to the trigger_file
        trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")

        x += 1

    return trigger_file


def _write_triggers_from_sequence_free_spell(array, trigger_file):
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


def trigger_decoder(mode, trigger_loc=None):

    # Load triggers.txt
    if not trigger_loc:
        trigger_loc = load_txt_data()

    with open(trigger_loc, 'r') as text_file:
        # Get every line of trigger.txt if that line does not contain
        # 'fixation'
        # [['words', 'in', 'line'], ['second', 'line']...]

        # trigger file has three columns: SYMBOL, TARGETNESS_INFO, TIMING

        trigger_txt = [line.split() for line in text_file if 'fixation' not in line and '+' not in line]

    # If operating mode is calibration, trigger.txt has three columns.
    if mode == 'calibration' or mode == 'copy_phrase':
        symbol_info = map(lambda x: x[0],trigger_txt)
        trial_target_info = map(lambda x: x[1],trigger_txt)
        timing_info = map(lambda x: eval(x[2]),trigger_txt)
    elif mode == 'free_spell':
        symbol_info = map(lambda x: x[0],trigger_txt)
        trial_target_info = None
        timing_info = map(lambda x: eval(x[1]),trigger_txt)
    else:
        raise Exception("You have not provided a valid operating mode for trigger_decoder. "
                        "Valid modes are: 'calibration','copy_phrase','free_spell'")

    return symbol_info, trial_target_info, timing_info
