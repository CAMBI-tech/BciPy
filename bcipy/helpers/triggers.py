from bcipy.helpers.load import load_txt_data
from bcipy.helpers.stimuli import resize_image, play_sound
import csv
from typing import TextIO, List, Tuple

from psychopy import visual, core

NONE_VALUE = '0'
SOUND_TYPE = 'sound'
IMAGE_TYPE = 'image'


class TriggerCallback:
    timing = None
    first_time = True

    def callback(self, clock: core.Clock, stimuli: str) -> None:
        if self.first_time:
            self.timing = [stimuli, clock.getTime()]
            self.first_time = False

    def reset(self):
        self.timing = None
        self.first_time = True


def _calibration_trigger(experiment_clock: core.Clock,
                         trigger_type: str='sound',
                         trigger_name: str='calibration_trigger',
                         display=None,
                         on_trigger=None) -> List[tuple]:
    """Calibration Trigger.

        Outputs triggers for the purpose of calibrating data and stimuli.
        This is an ongoing difficulty between OS, DAQ devices and stimuli type. This
        code aims to operationalize the approach to finding the correct DAQ samples in
        relation to our trigger code.

    PARAMETERS
    ---------
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
    trigger_callback = TriggerCallback()

    # If sound trigger is selected, output calibration tones
    if trigger_type == SOUND_TYPE:
        import sounddevice as sd
        import soundfile as sf

        play_sound(
            sound_file_path='bcipy/static/sounds/1k_800mV_20ms_stereo.wav',
            dtype='float32',
            track_timing=True,
            sound_callback=trigger_callback,
            sound_load_buffer_time=0.5,
            experiment_clock=experiment_clock,
            trigger_name='calibration_trigger')

    elif trigger_type == IMAGE_TYPE:
        if display:
            calibration_box = visual.ImageStim(
                win=display,
                image='bcipy/static/images/testing_images/white.png',
                pos=(-.5, -.5),
                mask=None,
                ori=0.0)
            calibration_box.size = resize_image(
                'bcipy/static/images/testing_images/white.png',
                display.size, 0.75)

            display.callOnFlip(trigger_callback.callback, experiment_clock, trigger_name)
            if on_trigger is not None:
                display.callOnFlip(on_trigger, trigger_name)

            presentation_time = int(1 * display.getActualFrameRate())
            for num_frames in range(presentation_time):
                calibration_box.draw()
                display.flip()

        else:
            raise Exception(
                'Display object required for calibration with images!')

    else:
        raise Exception('Trigger type not implemented for Calibration yet!')

    return trigger_callback.timing


def _write_triggers_from_sequence_calibration(
        array: list,
        trigger_file: TextIO,
        offset: bool=False):
    """Write triggers from calibration.

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


def _write_triggers_from_sequence_copy_phrase(
        array,
        trigger_file,
        copy_text,
        typed_text,
        offset=None):
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
        last_typed = typed_text[-1] if typed_text else None
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


def write_triggers_from_sequence_icon_to_icon(
        sequence_timing: List[Tuple], trigger_file: TextIO, target: str,
        target_displayed: bool, offset=None):
    """
    Write triggers from icon to icon task.
    It writes in the following order:
        (I) presented letter, (II) targetness, (III) timestamp

    Parameters:
    ----------
        sequence_timing - list of (icon, time) output from rsvp after
            displaying a sequence.
        trigger_file - open file in which to write.
        target - target for the current sequence
        target_displayed - whether or not the target was presented during the
            sequence.
    """
    if offset:
        (letter, time) = sequence_timing
        targetness = 'offset_correction'
        trigger_file.write('%s %s %s' % (letter, targetness, time) + "\n")
        return

    icons, _times = zip(*sequence_timing)
    calib_presented = 'calibration_trigger' in icons
    calib_index = 0 if calib_presented else -1

    if calib_presented:
        target_pres_index = 1
        fixation_index = 2
    elif target_displayed:
        target_pres_index = 0
        fixation_index = 1
    else:
        target_pres_index = -1
        fixation_index = 0

    for i, (icon, presentation_time) in enumerate(sequence_timing):
        targetness = 'nontarget'
        if i == calib_index:
            targetness = 'calib'
        elif i == target_pres_index:
            targetness = 'first_pres_target'
        elif i == fixation_index:
            targetness = 'fixation'
        elif icon == target:
            targetness = 'target'
        else:
            targetness = 'nontarget'
        trigger_file.write('%s %s %s' % (icon, targetness, presentation_time) +
                           "\n")


def trigger_decoder(mode: str, trigger_path: str=None) -> tuple:
    """Trigger Decoder.

    Given a mode of operation (calibration, copy phrase, etc) and
        a path to the trigger location (*.txt file), this function
        will split into symbols (A, ..., Z), timing info (32.222), and
        targetness (target, nontarget). It will also extract any saved
        offset information and pass that back.

    PARAMETERS
    ----------
    :param: mode: mode of bci operation. Note the mode changes how triggers
        are saved.
    :param: trigger_path: [Optional] path to triggers.txt file
    :return: tuple: symbol_info, trial_target_info, timing_info, offset.
    """

    # Load triggers.txt
    if not trigger_path:
        trigger_path = load_txt_data()

    # Get every line of trigger.txt
    with open(trigger_path, 'r+') as text_file:
        # most trigger files has three columns:
        #   SYMBOL, TARGETNESS_INFO[OPTIONAL], TIMING
        trigger_txt = [line.split() for line in text_file]

    # extract stimuli from the text
    stimuli_triggers = [line for line in trigger_txt
                        if line[1] == 'target' or
                        line[1] == 'nontarget']

    # from the stimuli array, pull our the symbol information
    symbol_info = list(map(lambda x: x[0], stimuli_triggers))

    # If operating mode is free spell, it only has 2 columns
    #   otherwise, it has 3
    if mode != 'free_spell':
        trial_target_info = list(map(lambda x: x[1], stimuli_triggers))
        timing_info = list(map(lambda x: eval(x[2]), stimuli_triggers))
    else:
        trial_target_info = None
        timing_info = list(map(lambda x: eval(x[1]), stimuli_triggers))

    # Get any offset or calibration triggers
    offset_array = [line[2] for line in trigger_txt
                    if line[0] == 'offset']
    calib_trigger_array = [line[2] for line in trigger_txt
                           if line[0] == 'calibration_trigger']

    # If present, calculate the offset between the DAQ and Triggers from display
    if len(offset_array) == 1 and len(calib_trigger_array) == 1:

        # Extract the offset and calibration trigger time
        offset_time = float(offset_array[0])
        calib_trigger_time = float(calib_trigger_array[0])

        # Calculate the offset (ASSUMES DAQ STARTED FIRST!)
        offset = offset_time - calib_trigger_time

    # Otherwise, assume no observed offset
    else:
        offset = 0

    return symbol_info, trial_target_info, timing_info, offset


class Labeller(object):
    """Labels the targetness for a trigger value in a raw_data file."""

    def __init__(self):
        super(Labeller, self).__init__()

    def label(self, trigger):
        raise NotImplementedError('Subclass must define the label method')


class LslCalibrationLabeller(Labeller):
    """Calculates targetness for calibration data. Uses a state machine to
    determine how to label triggers.

    Parameters:
    -----------
        seq_len: stim_length parameter value for the experiment; used to calculate
            targetness for first_pres_target.
    """

    def __init__(self, seq_len: int):
        super(LslCalibrationLabeller, self).__init__()
        self.seq_len = seq_len
        self.prev = None
        self.current_target = None
        self.seq_position = 0

    def label(self, trigger):
        """Calculates the targetness for the given trigger, accounting for the
        previous triggers/states encountered."""
        state = ''
        if self.prev is None:
            # First trigger is always calibration.
            state = 'calib'
        elif self.prev == 'calib':
            self.current_target = trigger
            state = 'first_pres_target'
        elif self.prev == 'first_pres_target':
            # reset the sequence when fixation '+' is encountered.
            self.seq_pos = 0
            state = 'fixation'
        else:
            self.seq_pos += 1
            if self.seq_pos > self.seq_len:
                self.current_target = trigger
                state = 'first_pres_target'
            elif trigger == self.current_target:
                state = 'target'
            else:
                state = 'nontarget'
        self.prev = state
        return state


class LslCopyPhraseLabeller(Labeller):
    """Sequentially calculates targetness for copy phrase triggers."""

    def __init__(self, copy_text: str, typed_text: str):
        super(LslCopyPhraseLabeller, self).__init__()
        self.copy_text = copy_text
        self.typed_text = typed_text
        self.prev = None

        self.pos = 0
        self.typing_pos = -1  # sequence length should be >= typed text.

        self.current_target = None

    def label(self, trigger):
        """Calculates the targetness for the given trigger, accounting for the
        previous triggers/states encountered."""
        state = ''
        if self.prev is None:
            state = 'calib'
        elif trigger == '+':
            self.typing_pos += 1
            if not self.current_target:
                # set target to first letter in the copy phrase.
                self.current_target = self.copy_text[self.pos]
            else:
                last_typed = self.typed_text[self.typing_pos - 1]
                if last_typed == self.current_target:
                    # increment if the user typed the target correctly
                    if last_typed != '<':
                        self.pos += 1
                    self.current_target = self.copy_text[self.pos]
                else:
                    # Error correction.
                    self.current_target = '<'

            state = 'fixation'
        else:
            if trigger == self.current_target:
                state = 'target'
            else:
                state = 'nontarget'
        self.prev = state
        return state


def _extract_triggers(csvfile: TextIO,
                      trg_field,
                      labeller: Labeller) -> List[Tuple[str, str, str]]:
    """Extracts trigger data from an experiment output csv file.
    Parameters:
    -----------
        csvfile: open csv file containing data.
        trg_field: optional; name of the data column with the trigger data;
                   defaults to 'TRG'
        labeller: Labeller used to calculate the targetness value for a
            given trigger.
    Returns:
    --------
        list of tuples of (trigger, targetness, timestamp)
    """
    data = []

    # Skip metadata rows
    _daq_type = next(csvfile)
    _sample_rate = next(csvfile)

    reader = csv.DictReader(csvfile)

    for row in reader:
        trg = row[trg_field]
        if trg != NONE_VALUE:
            if 'calibration' in trg:
                trg = 'calibration_trigger'
            targetness = labeller.label(trg)
            data.append((trg, targetness, row['timestamp']))

    return data


def write_trigger_file_from_lsl_calibration(csvfile: TextIO,
                                            trigger_file: TextIO,
                                            seq_len: int, trg_field: str='TRG'):
    """Creates a triggers.txt file from TRG data recorded in the raw_data
    output from a calibration."""
    extracted = extract_from_calibration(csvfile, seq_len, trg_field)
    _write_trigger_file_from_extraction(trigger_file, extracted)


def write_trigger_file_from_lsl_copy_phrase(csvfile: TextIO,
                                            trigger_file: TextIO,
                                            copy_text: str, typed_text: str,
                                            trg_field: str='TRG'):
    """Creates a triggers.txt file from TRG data recorded in the raw_data
    output from a copy phrase."""
    extracted = extract_from_copy_phrase(csvfile, copy_text, typed_text,
                                         trg_field)
    _write_trigger_file_from_extraction(trigger_file, extracted)


def _write_trigger_file_from_extraction(trigger_file: TextIO,
                                        extraction: List[Tuple[str, str, str]]):
    """Writes triggers that have been extracted from a raw_data file to a
    file."""
    for trigger, targetness, timestamp in extraction:
        trigger_file.write(f"{trigger} {targetness} {timestamp}\n")

    # TODO: is this assumption correct?
    trigger_file.write("offset offset_correction 0.0")


def extract_from_calibration(csvfile: TextIO,
                             seq_len: int,
                             trg_field: str='TRG') -> List[Tuple[str, str, str]]:
    """Extracts trigger data from a calibration output csv file.
    Parameters:
    -----------
        csvfile: open csv file containing data.
        seq_len: stim_length parameter value for the experiment; used to calculate
                 targetness for first_pres_target.
        trg_field: optional; name of the data column with the trigger data;
                   defaults to 'TRG'
    Returns:
    --------
        list of tuples of (trigger, targetness, timestamp), where timestamp is
        the timestamp recorded in the file.
    """

    return _extract_triggers(csvfile, trg_field,
                             labeller=LslCalibrationLabeller(seq_len))


def extract_from_copy_phrase(csvfile: TextIO,
                             copy_text: str,
                             typed_text: str,
                             trg_field: str='TRG') -> List[Tuple[str, str, str]]:
    """Extracts trigger data from a copy phrase output csv file.
    Parameters:
    -----------
        csvfile: open csv file containing data.
        copy_text: phrase to copy
        typed_text: participant typed response
        trg_field: optional; name of the data column with the trigger data;
                   defaults to 'TRG'
    Returns:
    --------
        list of tuples of (trigger, targetness, timestamp), where timestamp is
        the timestamp recorded in the file.
    """
    labeller = LslCopyPhraseLabeller(copy_text, typed_text)
    return _extract_triggers(csvfile, trg_field, labeller=labeller)
