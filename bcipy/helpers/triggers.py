# -*- coding: utf-8 -*-
from bcipy.helpers.load import load_txt_data


def _calibration_trigger(experiment_clock, trigger_type='sound', display=None,
                         on_trigger=None):
    """Calibration Trigger.

        Outputs triggers for the purpose of calibrating data and stimuli.
        This is an ongoing difficulty between OS, DAQ devices and stimuli type. This
        code aims to operationalize the approach to finding the correct DAQ samples in
        relation to our trigger code.

        Args:
                experiment_clock(clock): clock with getTime() method, which is used in the code
                    to report timing of stimuli
                trigger_type(string): type of trigger that is desired (sound, image, etc)
                display(DisplayWindow): a window that can display stimuli. Currently, a Psychopy window.
                on_trigger(function): optional callback; if present gets called
                         when the calibration trigger is fired; accepts a single
                         parameter for the timing information.
        Return:
                timing(array): timing values for the calibration triggers to be written to trigger file or
                    used to calculate offsets.
    """

    # If sound trigger is selected, output calibration tones
    if trigger_type == 'sound':
        import sounddevice as sd
        import soundfile as sf
        from psychopy import core

        # Init the sound object and give it some time to buffer
        try:
            data, fs = sf.read(
                'bcipy/static/sounds/1k_800mV_20ms_stereo.wav', dtype='float32')
        except:
            raise Exception('Sound object could not be found or Initialized')

        core.wait(.2)

        # Play the fist sound (used to calibrate) and wait.
        sd.play(data, fs)
        timing = ['calibration_trigger', experiment_clock.getTime()]
        if on_trigger:
            on_trigger(timing)
        core.wait(.3)

        # Play another to have active control over sound latency
        sd.play(data, fs)
        core.wait(.3)

    elif trigger_type == 'image':
        if display:
            from psychopy import visual, core

            calibration_box = visual.ImageStim(
                display,
                image='bcipy/static/images/testing_images/white.png',
                size=(.75, .75),
                pos=(-.5, -.5),
                mask=None,
                ori=0.0)
            timing = ['calibration_trigger', experiment_clock.getTime()]
            if on_trigger:
                on_trigger(timing)

            calibration_box.draw()
            display.flip()

            core.wait(1)

            # delete the stimuli off screen
            display.flip()
            core.wait(.1)

            # Draw it again and flip
            calibration_box.draw()
            display.flip()

            core.wait(1)

        else:
            raise Exception(
                'No Display Object passed for calibration with images!')

    else:
        raise Exception('Trigger type not implemented for Calibration yet!')

    return timing


def _write_triggers_from_sequence_calibration(array, trigger_file, offset=None):
    """
    Write triggers from calibration.

    Helper Function to write trigger data to provided trigger_file. It assigns
        target letter based on the first presented letter in sequence, then
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


def _write_triggers_from_sequence_copy_phrase(array, trigger_file,
                                              copy_text, typed_text, offset=None):
    """
    Write triggers from copy phrase.

    Helper Function to write trigger data to provided trigger_file. It assigns
        target letter based on matching the next needed letter in typed text
        then assigns target/nontarget label to following letters.

    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp
    """

    if offset:
        # extract the letter and timing from the array
        (letter, time) = array
        targetness = 'offset_correction'
        trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")
    else:
        # get relevant spelling info to determine what was and should be typed
        spelling_length = len(typed_text)
        last_typed = typed_text[-1]
        correct_letter = copy_text[spelling_length - 1]

        # because there is the impassibility of incorrect letter and correction,
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
        # 'fixation' or 'offset_correction'
        # [['words', 'in', 'line'], ['second', 'line']...]

        # trigger file has three columns: SYMBOL, TARGETNESS_INFO, TIMING

        trigger_txt = [line.split() for line in text_file if 'fixation' not in line and '+' not in line
                       and 'offset_correction' not in line and 'calibration_trigger' not in line]

    # If operating mode is calibration, trigger.txt has three columns.
    if mode == 'calibration' or mode == 'copy_phrase':
        symbol_info = list(map(lambda x: x[0], trigger_txt))
        trial_target_info = list(map(lambda x: x[1], trigger_txt))
        timing_info = list(map(lambda x: eval(x[2]), trigger_txt))
    elif mode == 'free_spell':
        symbol_info = list(map(lambda x: x[0], trigger_txt))
        trial_target_info = None
        timing_info = list(map(lambda x: eval(x[1]), trigger_txt))
    else:
        raise Exception("You have not provided a valid operating mode for trigger_decoder. "
                        "Valid modes are: 'calibration','copy_phrase','free_spell'")

    with open(trigger_loc, 'r') as text_file:
        offset_array = [line.split()
                        for line in text_file if 'offset_correction' in line]

    if offset_array:
        with open(trigger_loc, 'r') as text_file:
            calib_trigger_time = [
                line.split() for line in text_file if 'calibration_trigger' in line]

        if calib_trigger_time:
            if offset_array:
                offset = float(
                    calib_trigger_time[0][2]) - float(offset_array[0][2])
            else:
                offset = timing_info[0] - float(offset_array[0][2])
    else:
        offset = 0

    return symbol_info, trial_target_info, timing_info, offset
