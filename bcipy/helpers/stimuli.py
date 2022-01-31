import glob
import itertools
import re
import logging
import random
from os import path, sep
from typing import Iterator, List, Tuple, NamedTuple

import numpy as np
from enum import Enum

import sounddevice as sd
import soundfile as sf
from PIL import Image

from bcipy.helpers.exceptions import BciPyCoreException

from psychopy import core

# Prevents pillow from filling the console with debug info
logging.getLogger('PIL').setLevel(logging.WARNING)
log = logging.getLogger(__name__)
DEFAULT_FIXATION_PATH = 'bcipy/static/images/main/PLUS.png'


class StimuliOrder(Enum):
    """Stimuli Order.

    Enum to define the ordering of stimuli for inquiry.
    """
    RANDOM = 'random'
    ALPHABETICAL = 'alphabetical'

    @classmethod
    def list(cls):
        """Returns all enum values as a list"""
        return list(map(lambda c: c.value, cls))


class InquirySchedule(NamedTuple):
    """Schedule for the next inquiries to present, where each inquiry specifies
    the stimulus, duration, and color information.

    Attributes
    ----------
    - stimuli: `List[List[str]]`
    - durations: `List[List[float]]`
    - colors: `List[List[str]]`
    """
    stimuli: List[List[str]]
    durations: List[List[float]]
    colors: List[List[str]]


def alphabetize(stimuli: List[str]) -> List[str]:
    """Alphabetize.

    Given a list of string stimuli, return a list of sorted stimuli by alphabet.
    """
    return sorted(stimuli, key=lambda x: re.sub(r'[^a-zA-Z0-9 \n\.]', 'ZZ', x).lower())


def inq_generator(query: List[str],
                  timing: List[float] = [1, 0.2],
                  color: List[str] = ['red', 'white'],
                  inquiry_count: int = 1,
                  stim_order: StimuliOrder = StimuliOrder.RANDOM,
                  is_txt: bool = True) -> InquirySchedule:
    """ Given the query set, prepares the stimuli, color and timing
        Args:
            query(list[str]): list of queries to be shown
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
            """

    if stim_order == StimuliOrder.ALPHABETICAL:
        query = alphabetize(query)
    else:
        random.shuffle(query)

    stim_length = len(query)

    # Init some lists to construct our stimuli with
    samples, times, colors = [], [], []
    for _ in range(inquiry_count):

        # append a fixation cross. if not text, append path to image fixation
        sample = [get_fixation(is_txt)]

        # construct the sample from the query
        sample += [i for i in query]
        samples.append(sample)

        # append timing
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * stim_length)

        # append colors
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * stim_length)
    return InquirySchedule(samples, times, colors)


def best_selection(selection_elements: list,
                   val: list,
                   len_query: int,
                   always_included: List[str] = None) -> list:
    """Best Selection.

     given set of elements and a value function over the set, picks the len_query
        number of elements with the best value.

        Args:
            selection_elements(list[str]): the set of elements
            val(list[float]): values for the corresponding elements
            len_query(int): number of elements to be picked from the set
            always_included(list[str]): subset of elements that should always be
                included in the result. Defaults to None.
        Return:
            best_selection(list[str]): elements from selection_elements with the best values """

    always_included = always_included or []
    # pick the top n items sorted by value in decreasing order
    elem_val = dict(zip(selection_elements, val))
    best = sorted(selection_elements, key=elem_val.get, reverse=True)[0:len_query]

    replacements = [
        item for item in always_included
        if item not in best and item in selection_elements
    ][0:len_query]

    if replacements:
        best[-len(replacements):] = replacements
    return best


def best_case_rsvp_inq_gen(alp: list,
                           session_stimuli: np.ndarray,
                           timing: List[float] = [1, 0.2],
                           color: List[str] = ['red', 'white'],
                           stim_number: int = 1,
                           stim_length: int = 10,
                           stim_order: StimuliOrder = StimuliOrder.RANDOM,
                           is_txt: bool = True,
                           inq_constants: List[str] = None) -> InquirySchedule:
    """Best Case RSVP Inquiry Generation.

    generates RSVPKeyboard inquiry by picking n-most likely letters.
        Args:
            alp(list[str]): alphabet (can be arbitrary)
            session_stimuli(ndarray[float]): quantifier metric for query selection
                dim(session_stimuli) = card(alp)!
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            stim_number(int): number of random stimuli to be created
            stim_length(int): number of trials in a inquiry
            stim_order(StimuliOrder): ordering of stimuli in the inquiry
            inq_constants(list[str]): list of letters that should always be
                included in every inquiry. If provided, must be alp items.
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
            """

    if len(alp) != len(session_stimuli):
        raise BciPyCoreException((
            f'Missing information about alphabet.'
            f'len(alp):{len(alp)} and len(session_stimuli):{len(session_stimuli)} should be same!'))

    if inq_constants and not set(inq_constants).issubset(alp):
        raise BciPyCoreException('Inquiry constants must be alphabet items.')

    # query for the best selection
    query = best_selection(
        alp,
        session_stimuli,
        stim_length,
        inq_constants)

    if stim_order == StimuliOrder.ALPHABETICAL:
        query = alphabetize(query)
    else:
        random.shuffle(query)

    # Init some lists to construct our stimuli with
    samples, times, colors = [], [], []
    for _ in range(stim_number):

        # append a fixation cross. if not text, append path to image fixation
        sample = [get_fixation(is_txt)]

        # construct the sample from the query
        sample += [i for i in query]
        samples.append(sample)

        # append timing
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * stim_length)

        # append colors
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * stim_length)
    return InquirySchedule(samples, times, colors)


def calibration_inquiry_generator(
        alp: List[str],
        timing: List[float] = [0.5, 1, 0.2],
        color: List[str] = ['green', 'red', 'white'],
        stim_number: int = 10,
        stim_length: int = 10,
        stim_order: StimuliOrder = StimuliOrder.RANDOM,
        is_txt: bool = True) -> InquirySchedule:
    """Random Calibration Inquiry Generator.

    Generates random inquiries with target letters in all possible positions.
        Args:
            alp(list[str]): stimuli
            timing(list[float]): Task specific timing for generator.
                [target, fixation, stimuli]
            color(list[str]): Task specific color for generator
                [target, fixation, stimuli]
            stim_number(int): number of trials in a inquiry
            stim_length(int): number of random stimuli to be created
            stim_order(StimuliOrder): ordering of stimuli in the inquiry
            is_txt(bool): whether or not the stimuli type is text. False would be an image stimuli.
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
    """

    len_alp = len(alp)

    targets = distributed_target_positions(stim_number, stim_length)

    samples, times, colors = [], [], []
    for _ in range(stim_number):
        idx = np.random.permutation(np.array(list(range(len_alp))))
        #take random sample of stim_length (+1 for no target inquiries)
        rand_smp = (idx[0:stim_length + 1])

        # define the target and fixation that come before an inquiry
        if not is_txt:
            sample = [
                alp[rand_smp[0]],
                get_fixation(is_txt=False)]
        else:
            #define target using distributed target indexes
            sample = [alp[rand_smp[targets[0]]], '+']
            targets = targets[1:]

        # shuffle the samples using the permutated random indexes
        #rand_smp = np.random.permutation(rand_smp)

        #cut off extra letter used for no target inquiries
        rand_smp = rand_smp[0:stim_length]
        if stim_order == StimuliOrder.ALPHABETICAL:
            inquiry = alphabetize([alp[i] for i in rand_smp])
        else:
            inquiry = [alp[i] for i in rand_smp]
        sample.extend(inquiry)
        samples.append(sample)
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * stim_length)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * stim_length)

    return InquirySchedule(samples, times, colors)

def distributed_target_positions(stim_number: int, stim_length: int) -> list:
    """Distributed Target Positions.

    Generates evenly distributed target positions, including no target position, and shuffles them.
    Args:
        stim_number(int): Number of trials for the experiment
        stim_length(int): Number of stimuli in each inquiry

    Return distributed_target_positions(list): targets: array of target indexes to be chosen
    """

    target_positions = stim_length + 1
    num_target_pos = (int) (stim_number/target_positions)
    num_rem_pos = (stim_number%target_positions)
    targets = []

    count = 0
    while(count<target_positions):
        for _ in range(num_target_pos):
            targets.append(count)
        count = count + 1

    rem_pos = np.random.permutation(np.array(list(range(num_target_pos))))
    rem_pos = (rem_pos[0:num_rem_pos])

    targets.extend(rem_pos)
    targets = np.random.permutation(targets)

    return targets

def get_task_info(experiment_length: int, task_color: str) -> Tuple[List[str], List[str]]:
    """Get Task Info.

    Generates fixed RSVPKeyboard task text and color information for
            display.
    Args:
        experiment_length(int): Number of inquiries for the experiment
        task_color(str): Task information display color

    Return get_task_info((tuple): task_text: array of task text to display
                   task_color: array of colors for the task text
                   )
    """

    # Do list comprehensions to get the arrays for the task we need.
    task_text = ['%s/%s' % (stim + 1, experiment_length)
                 for stim in range(experiment_length)]
    task_color = [[str(task_color)] for stim in range(experiment_length)]

    return (task_text, task_color)


def resize_image(image_path: str, screen_size: tuple, sti_height: int) -> Tuple[float, float]:
    """Resize Image.

    Returns the width and height that a given image should be displayed at
    given the screen size, size of the original image, and stimuli height
    parameter"""
    # Retrieve image width and height
    with Image.open(image_path) as pillow_image:
        image_width, image_height = pillow_image.size

    # Resize image so that its largest dimension is the stimuli size defined
    # in the parameters file
    if image_width >= image_height:
        proportions = (1, (image_height / image_width))
    else:
        proportions = ((image_width / image_height), 1)

    # Adjust image size to scale with monitor size
    screen_width, screen_height = screen_size
    if screen_width >= screen_height:
        sti_size = ((screen_height / screen_width) * sti_height *
                    proportions[0], sti_height * proportions[1])
    else:
        sti_size = (
            sti_height * proportions[0],
            (screen_width / screen_height) * sti_height * proportions[1])

    return sti_size


def play_sound(sound_file_path: str,
               dtype: str = 'float32',
               track_timing: bool = False,
               sound_callback=None,
               sound_load_buffer_time: float = 0.5,
               sound_post_buffer_time: float = 1,
               experiment_clock=None,
               trigger_name: str = None,
               timing: list = []) -> list:
    """Play Sound.

    Using soundevice and soundfile, play a sound giving options to buffer times between
        loading sound into memory and after playing. If desired, marker writers or list based
        timing with psychopy clocks may be passed and sound timing returned.


    PARAMETERS
    ----------
    :param: sound_file_path
    :param: dtype: type of sound ex. float32.
    :param: track_timing: whether or not to track timing of sound playin
    :param: sound_callback: trigger based callback (see MarkerWriter and NullMarkerWriter)
    :param: sound_load_buffer_time: time to wait after loading file before playing
    :param: sound_post_buffer_time: time to wait after playing sound before returning
    :param: experiment_clock: psychopy clock to get time of sound stimuli
    :param: trigger_name: name of the sound trigger
    :param: timing: list of triggers in the form of trigger name, trigger timing
    :resp: timing
    """

    try:
        # load in the sound file and wait some time before playing
        data, fs = sf.read(sound_file_path, dtype=dtype)
        core.wait(sound_load_buffer_time)

    except Exception as e:
        error_message = f'Sound file could not be found or initialized. \n Exception={e}'
        log.exception(error_message)
        raise BciPyCoreException(error_message)

    #  if timing is wanted, get trigger timing for this sound stimuli
    if track_timing:
        # if there is a timing callback for sound, evoke it
        if sound_callback is not None:
            sound_callback(experiment_clock, trigger_name)
        timing.append([trigger_name, experiment_clock.getTime()])

    # play our loaded sound and wait for some time before it's finished
    # NOTE: there is a measurable delay for calling sd.play. (~ 0.1 seconds;
    # which I believe happens prior to the sound playing).
    sd.play(data, fs)
    if sound_post_buffer_time:
        # sd.play returns immediately (according to the docs); calculate offset
        # so the sound_post_buffer_time accounts for the duration of the sound.
        duration = len(data) / fs
        core.wait(sound_post_buffer_time + duration)
    return timing


def soundfiles(directory: str) -> Iterator[str]:
    """Creates a generator that cycles through sound files (.wav) in a
    directory and returns the path to next sound file on each iteration.

    Parameters:
    -----------
        directory - path to the directory which contains .wav files
    Returns:
    --------
        iterator that infinitely cycles through the filenames.
    """
    if not path.isdir(directory):
        error_message = f'Invalid directory=[{directory}] for sound files.'
        log.error(error_message)
        raise BciPyCoreException(error_message)
    if not directory.endswith(sep):
        directory += sep
    return itertools.cycle(glob.glob(directory + '*.wav'))


def get_fixation(is_txt: bool) -> str:
    """Get Fixation.

    Return the correct stimulus fixation given the type (text or image).
    """
    if is_txt:
        return '+'
    else:
        return DEFAULT_FIXATION_PATH
