# -*- coding: utf-8 -*-


def _write_triggers_from_sequence_calibration(array, file):

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

        # write to the file
        file.write('%s %s %s' % (letter, targetness, time) + "\n")

        x += 1

    return file


def _write_triggers_from_sequence_copy_phrase(array, file,
                                              copy_text, typed_text):

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
            targetness = 'correct'
        else:
            targetness = 'incorrect'

        # write to the file
        file.write('%s %s %s' % (letter, targetness, time) + "\n")

        x += 1

    return file


def _write_triggers_from_sequence_free_spell(array, file):

    x = 0

    for i in array:

        # extract the letter and timing from the array
        (letter, time) = i

        # write to file
        file.write('%s %s' % (letter, time) + "\n")

        x += 1

    return file
