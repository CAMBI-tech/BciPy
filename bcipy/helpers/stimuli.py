import glob
import itertools
import logging
import random
from os import path, sep
from typing import Iterator, List, NamedTuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from PIL import Image
from psychopy import core

# Prevents pillow from filling the console with debug info
logging.getLogger("PIL").setLevel(logging.WARNING)


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


def rsvp_inq_generator(query: list,
                       timing=[1, 0.2],
                       color=['red', 'white'],
                       stim_number=1,
                       is_txt=True,
                       inq_constants=None) -> InquirySchedule:
    """ Given the query set, prepares the stimuli, color and timing
        Args:
            query(list[str]): list of queries to be shown
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            inq_constants(list[str]): list of letters that should always be
                included in every inquiry. If provided, must be alp items.
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
            """

    # shuffle the returned values
    random.shuffle(query)

    stim_length = len(query)

    # Init some lists to construct our stimuli with
    samples, times, colors = [], [], []
    for idx_num in range(stim_number):

        # append a fixation cross. if not text, append path to image fixation
        if is_txt:
            sample = ['+']
        else:
            sample = ['bcipy/static/images/bci_main_images/PLUS.png']

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
                   always_included=None) -> list:
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
    n = len_query
    # pick the top n items sorted by value in decreasing order
    elem_val = dict(zip(selection_elements, val))
    best = sorted(selection_elements, key=elem_val.get, reverse=True)[0:n]

    replacements = [
        item for item in always_included
        if item not in best and item in selection_elements
    ][0:n]

    if replacements:
        best[-len(replacements):] = replacements
    return best


def best_case_rsvp_inq_gen(alp: list,
                           session_stimuli: list,
                           timing=[1, 0.2],
                           color=['red', 'white'],
                           stim_number=1,
                           stim_length=10,
                           is_txt=True,
                           inq_constants=None) -> InquirySchedule:
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
            inq_constants(list[str]): list of letters that should always be
                included in every inquiry. If provided, must be alp items.
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
            """

    if len(alp) != len(session_stimuli):
        raise Exception('Missing information about alphabet. len(alp):{}, '
                        'len(session_stimuli):{}, should be same!'.format(
                            len(alp), len(session_stimuli)))

    if inq_constants and not set(inq_constants).issubset(alp):
        raise Exception('Inquiry constants must be alphabet items.')

    # create a list of alphabet letters
    alphabet = [i for i in alp]

    # query for the best selection
    query = best_selection(
        alphabet,
        session_stimuli,
        stim_length,
        inq_constants)

    # shuffle the returned values
    random.shuffle(query)

    # Init some lists to construct our stimuli with
    samples, times, colors = [], [], []
    for idx_num in range(stim_number):

        # append a fixation cross. if not text, append path to image fixation
        if is_txt:
            sample = ['+']
        else:
            sample = ['bcipy/static/images/bci_main_images/PLUS.png']

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


def random_rsvp_calibration_inq_gen(alp,
                                    timing=[0.5, 1, 0.2],
                                    color=['green', 'red', 'white'],
                                    stim_number=10,
                                    stim_length=10,
                                    is_txt=True) -> InquirySchedule:
    """Random RSVP Calibration Inquiry Generator.

    Generates random RSVPKeyboard inquiries.
        Args:
            alp(list[str]): alphabet (can be arbitrary)
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            stim_number(int): number of random stimuli to be created
            stim_length(int): number of trials in a inquiry
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
    """

    len_alp = len(alp)

    samples, times, colors = [], [], []
    for idx_num in range(stim_number):
        idx = np.random.permutation(np.array(list(range(len_alp))))
        rand_smp = (idx[0:stim_length])
        if not is_txt:
            sample = [
                alp[rand_smp[0]],
                'bcipy/static/images/bci_main_images/PLUS.png']
        else:
            sample = [alp[rand_smp[0]], '+']
        rand_smp = np.random.permutation(rand_smp)
        sample += [alp[i] for i in rand_smp]
        samples.append(sample)
        times.append([timing[i] for i in range(len(timing) - 1)] +
                     [timing[-1]] * stim_length)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * stim_length)

    return InquirySchedule(samples, times, colors)


def target_rsvp_inquiry_generator(alp,
                                  target_letter,
                                  parameters,
                                  timing=[0.5, 1, 0.2],
                                  color=['green', 'white', 'white'],
                                  stim_length=10,
                                  is_txt=True) -> InquirySchedule:
    """Target RSVP Inquiry Generator.

    Generate target RSVPKeyboard inquiries.

        Args:
            alp(list[str]): alphabet (can be arbitrary)
            target_letter([str]): letter to be copied
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            stim_length(int): number of trials in a inquiry
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
    """

    len_alp = len(alp)

    # initialize our arrays
    samples, times, colors = [], [], []
    rand_smp = random.sample(range(len_alp), stim_length)
    if is_txt:
        sample = ['+']
    else:
        sample = ['bcipy/static/images/bci_main_images/PLUS.png']
        target_letter = parameters['path_to_presentation_images'] + \
            target_letter + '.png'
    sample += [alp[i] for i in rand_smp]

    # if the target isn't in the array, replace it with some random index that
    #  is not fixation
    if target_letter not in sample:
        random_index = np.random.randint(0, stim_length - 1)
        sample[random_index + 1] = target_letter

    # add target letter to start
    sample = [target_letter] + sample

    # to-do shuffle the target letter

    samples.append(sample)
    times.append([timing[i] for i in range(len(timing) - 1)] +
                 [timing[-1]] * stim_length)
    colors.append([color[i] for i in range(len(color) - 1)] +
                  [color[-1]] * stim_length)

    return InquirySchedule(samples, times, colors)


def get_task_info(experiment_length, task_color):
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


def rsvp_copy_phrase_inq_generator(alp, target_letter, timing=[0.5, 1, 0.2],
                                   color=['green', 'white', 'white'],
                                   stim_length=10) -> InquirySchedule:
    """Generate copy phrase RSVPKeyboard inquiries.

        Args:
            alp(list[str]): alphabet (can be arbitrary)
            target_letter([str]): letter to be copied
            timing(list[float]): Task specific timing for generator
            color(list[str]): Task specific color for generator
                First element is the target, second element is the fixation
                Observe that [-1] element represents the trial information
            stim_length(int): number of trials in an inquiry
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
    """

    len_alp = len(alp)

    # initialize our arrays
    samples, times, colors = [], [], []
    rand_smp = np.random.randint(0, len_alp, stim_length)
    sample = ['+']
    sample += [alp[i] for i in rand_smp]

    # if the target isn't in the array, replace it with some random index that
    #  is not fixation
    if target_letter not in sample:
        random_index = np.random.randint(0, stim_length - 1)
        sample[random_index + 1] = target_letter

    # to-do shuffle the target letter

    samples.append(sample)
    times.append([timing[i] for i in range(len(timing) - 1)] +
                 [timing[-1]] * stim_length)
    colors.append([color[i] for i in range(len(color) - 1)] +
                  [color[-1]] * stim_length)

    return InquirySchedule(samples, times, colors)


def generate_icon_match_images(
        experiment_length, image_path, number_of_inquiries, timing, is_word):
    """Generate Image Icon Matches.

    Generates an array of images to use for the icon matching task.
    Args:
        experiment_length(int): Number of images per inquiry
        image_path(str): Path to image files
        number_of_inquiries(int): Number of inquiries to generate
        timing(list): List of timings; [parameters['time_target'],
                       parameters['time_cross'],
                       parameters['time_flash']]
        is_word(bool): Whether or not this is an icon to word matching task
    Return generate_icon_match_images(arrays of tuples of paths to images to
    display, and timings)
    """
    # Get all png images in image path
    image_array = [
        img for img in glob.glob(image_path + "*.png")
        if not img.endswith("PLUS.png")
    ]

    if experiment_length > len(image_array) - 1:
        raise Exception(
            'Number of images to be displayed on screen is longer than number of images available')

    # Generate indexes of target images
    target_image_numbers = np.random.randint(
        0, len(image_array), number_of_inquiries)

    # Array of images to return
    return_array = []

    # Array of timings to return
    return_timing = []
    for specific_time in range(len(timing) - 1):
        return_timing.append(timing[specific_time])
    for item_without_timing in range(len(return_timing), experiment_length):
        return_timing.append(timing[-1])

    for inquiry in range(number_of_inquiries):
        return_array.append([])
        # Generate random permutation of image indexes
        random_number_array = np.random.permutation(len(image_array))
        if is_word:
            # Add name of target image to array
            image_path = path.basename(
                image_array[target_image_numbers[inquiry]])
            return_array[inquiry].append(image_path.replace('.png', ''))
        else:
            # Add target image to image array
            return_array[inquiry].append(
                image_array[target_image_numbers[inquiry]])
        # Add PLUS.png to image array TODO: get this from parameters file
        return_array[inquiry].append(
            'bcipy/static/images/bci_main_images/PLUS.png')

        # Add target image to inquiry, if it is not already there
        if not target_image_numbers[inquiry] in random_number_array[
                2:experiment_length]:
            random_number_array[np.random.randint(
                2, experiment_length)] = target_image_numbers[inquiry]

        # Fill the rest of the image array with random images
        for item in range(2, experiment_length):
            return_array[inquiry].append(
                image_array[random_number_array[item]])

    return (return_array, return_timing)


def resize_image(image_path: str, screen_size: tuple, sti_height: int):
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
        data, fs = sf.read(
            sound_file_path, dtype=dtype)
        core.wait(sound_load_buffer_time)

    except Exception:
        raise Exception(
            'StimGenPlaySoundError: sound file could not be found or initialized.')

    #  if timing is wanted, get trigger timing for this sound stimuli
    if track_timing:
        timing.append(trigger_name)
        timing.append(experiment_clock.getTime())

        # if there is a timing callback for sound, evoke it with the timing
        # list
        if sound_callback is not None:
            sound_callback(timing)

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
        raise Exception(('Invalid directory for sound files. Please check '
                         'your configuration.'))
    if not directory.endswith(sep):
        directory += sep
    return itertools.cycle(glob.glob(directory + '*.wav'))
