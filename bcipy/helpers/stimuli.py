import glob
import itertools
import logging
import random
import re

from abc import ABC, abstractmethod
from enum import Enum
from os import path, sep
from typing import Iterator, List, Tuple, NamedTuple, Optional

from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.list import grouper

from PIL import Image
# Prevents pillow from filling the console with debug info
logging.getLogger('PIL').setLevel(logging.WARNING)

from psychopy import core
import numpy as np
import sounddevice as sd
import soundfile as sf
from mne import Annotations, Epochs
from mne.io import RawArray
import mne


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


class TargetPositions(Enum):
    """Target Positions.

    Enum to define the positions of targets within the inquiry.
    """
    RANDOM = 'random'
    DISTRIBUTED = 'distributed'

    @classmethod
    def list(cls):
        """Returns all enum values as a list"""
        return list(map(lambda c: c.value, cls))


class PhotoDiodeStimuli(Enum):
    """Photodiode Stimuli.

    Enum to define unicode stimuli needed for testing system timing.
    """

    EMPTY = '\u25A1'  # box with a white border, no fill
    SOLID = '\u25A0'  # solid white box

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


class Reshaper(ABC):

    @abstractmethod
    def __call__(self):
        ...


class InquiryReshaper:
    def __call__(self,
                 trial_targetness_label: List[str],
                 timing_info: List[float],
                 eeg_data: np.ndarray,
                 sample_rate: int,
                 trials_per_inquiry: int,
                 offset: float = 0,
                 channel_map: List[int] = None,
                 poststimulus_length: float = 0.5,
                 prestimulus_length: float = 0.0,
                 transformation_buffer: float = 0.0,
                 target_label: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Extract inquiry data and labels.

        Args:
            trial_targetness_label (List[str]): labels each trial as "target", "non-target", "first_pres_target", etc
            timing_info (List[float]): Timestamp of each event in seconds
            eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
            sample_rate (int): sample rate of data provided in eeg_data
            trials_per_inquiry (int): number of trials in each inquiry
            offset (float, optional): Any calculated or hypothesized offsets in timings. Defaults to 0.
            channel_map (List[int], optional): Describes which channels to include or discard.
                 Defaults to None; all channels will be used.
            poststimulus_length (float, optional): time in seconds needed after the last trial in an inquiry.
            prestimulus_length (float, optional): time in seconds needed before the first trial in an inquiry.
            transformation_buffer (float, optional): time in seconds to buffer the end of the inquiry. Defaults to 0.0.
            target_label (str): label of target symbol. Defaults to "target"

        Returns:
            reshaped_data (np.ndarray): inquiry data of shape (Channels, Inquiries, Samples)
            labels (np.ndarray): integer label for each inquiry. With `trials_per_inquiry=K`,
                a label of [0, K-1] indicates the position of `target_label`, or label of K indicates
                `target_label` was not present.
            reshaped_trigger_timing (List[List[int]]): For each inquiry, a list of the sample index where each trial
                begins, accounting for the prestim buffer that may have been added to the front of each inquiry.
        """
        if channel_map:
            # Remove the channels that we are not interested in
            channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
            eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

        n_inquiry = len(timing_info) // trials_per_inquiry
        trial_duration_samples = int(poststimulus_length * sample_rate)
        prestimulus_samples = int(prestimulus_length * sample_rate)

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = list(map(lambda x: int((x + offset) * sample_rate), timing_info))

        # First, find the longest inquiry in this experiment
        # We'll add or remove a few samples from all other inquiries, to match this length
        def get_inquiry_len(inq_trigs):
            return inq_trigs[-1] - inq_trigs[0]

        longest_inquiry = max(grouper(triggers, trials_per_inquiry, fillvalue='x'), key=lambda xy: get_inquiry_len(xy))
        num_samples_per_inq = get_inquiry_len(longest_inquiry) + trial_duration_samples
        buffer_samples = int(transformation_buffer * sample_rate)

        # Label for every inquiry
        labels = np.zeros(
            (n_inquiry, trials_per_inquiry), dtype=np.compat.long
        )  # maybe this can be configurable? return either class indexes or labels ('nontarget' etc)
        reshaped_data, reshaped_trigger_timing = [], []
        for inquiry_idx, trials_within_inquiry in enumerate(
            grouper(zip(trial_targetness_label, triggers), trials_per_inquiry, fillvalue='x')
        ):
            first_trigger = trials_within_inquiry[0][1]

            trial_triggers = []
            for trial_idx, (trial_label, trigger) in enumerate(trials_within_inquiry):
                if trial_label == target_label:
                    labels[inquiry_idx, trial_idx] = 1

                # If presimulus buffer is used, we add it here so that trigger timings will
                # still line up with trial onset
                trial_triggers.append((trigger - first_trigger) + prestimulus_samples)
            reshaped_trigger_timing.append(trial_triggers)
            start = first_trigger - prestimulus_samples
            stop = first_trigger + num_samples_per_inq + buffer_samples
            reshaped_data.append(eeg_data[:, start:stop])

        return np.stack(reshaped_data, 1), labels, reshaped_trigger_timing

    @staticmethod
    def extract_trials(inquiries, samples_per_trial, inquiry_timing, downsample_rate=1):
        """Extract Trials.

        After using the InquiryReshaper, it may be necessary to futher trial the data for processing.
        Using the number of samples and inquiry timing, the data is reshaped from Channels, Inquiry, Samples to
        Channels, Trials, Samples. These should match with the trials extracted from the TrialReshaper given the same
        slicing parameters.
        """
        new_trials = []
        num_inquiries = inquiries.shape[1]
        for inquiry_idx, timing in zip(range(num_inquiries), inquiry_timing):  # C x I x S

            for time in timing:
                time = time // downsample_rate
                y = time + samples_per_trial
                new_trials.append(inquiries[:, inquiry_idx, time:y])
        return np.stack(new_trials, 1)  # C x T x S


class TrialReshaper(Reshaper):
    def __call__(self,
                 trial_targetness_label: list,
                 timing_info: list,
                 eeg_data: np.ndarray,
                 sample_rate: int,
                 offset: float = 0,
                 channel_map: List[int] = None,
                 poststimulus_length: float = 0.5,
                 prestimulus_length: float = 0.0,
                 target_label: str = "target") -> Tuple[np.ndarray, np.ndarray]:
        """Extract trial data and labels.

        Args:
            trial_targetness_label (list): labels each trial as "target", "non-target", "first_pres_target", etc
            timing_info (list): Timestamp of each event in seconds
            eeg_data (np.ndarray): shape (channels, samples) preprocessed EEG data
            sample_rate (int): sample rate of preprocessed EEG data
            trials_per_inquiry (int, optional): unused, kept here for consistent interface with `inquiry_reshaper`
            offset (float, optional): Any calculated or hypothesized offsets in timings.
                Defaults to 0.
            channel_map (List, optional): Describes which channels to include or discard.
                Defaults to None; all channels will be used.
            poststimulus_length (float, optional): [description]. Defaults to 0.5.
            target_label (str): label of target symbol. Defaults to "target"

        Returns:
            trial_data (np.ndarray): shape (channels, trials, samples) reshaped data
            labels (np.ndarray): integer label for each trial
        """
        # Remove the channels that we are not interested in
        if channel_map:
            channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
            eeg_data = np.delete(eeg_data, channels_to_remove, axis=0)

        # Number of samples we are interested per trial
        poststim_samples = int(poststimulus_length * sample_rate)
        prestim_samples = int(prestimulus_length * sample_rate)

        # triggers in seconds are mapped to triggers in number of samples.
        triggers = list(map(lambda x: int((x + offset) * sample_rate), timing_info))

        # Label for every trial in 0 or 1
        targetness_labels = np.zeros(len(triggers), dtype=np.compat.long)
        reshaped_trials = []
        for trial_idx, (trial_label, trigger) in enumerate(zip(trial_targetness_label, triggers)):
            if trial_label == target_label:
                targetness_labels[trial_idx] = 1

            # For every channel append filtered channel data to trials
            reshaped_trials.append(eeg_data[:, trigger - prestim_samples: trigger + poststim_samples])

        return np.stack(reshaped_trials, 1), targetness_labels


def mne_epochs(
        mne_data: RawArray,
        trigger_timing: List[float],
        trigger_labels: List[int],
        interval: Tuple[float, float],
        channels: Optional[List[str]] = None,
        detrend: Optional[int] = None,
        baseline: Optional[Tuple[float, float]] = (0, 0),
        preload: bool = True,
        reject_by_annotation: bool = False) -> Epochs:
    """MNE Epochs.

    Using an MNE RawArray, reshape the data given trigger information. If two labels present [0, 1],
    each may be accessed by numbered order. Ex. first_class = epochs['1'], second_class = epochs['2']

    Args:
        mne_data (RawArray): MNE RawArray object
        trigger_timing (List[float]): List of trigger timings in seconds
        trigger_labels (List[int]): List of trigger labels
        interval (Tuple[float, float]): Interval to extract data from trigger
        channels (Optional[List[str]], optional): List of channels to extract. Defaults to None.
        detrend (Optional[int]): Detrend order. Defaults to None.
        baseline (Optional[Tuple[float, float]]): Baseline interval to apply to epoch. Defaults to (0, 0).
        preload (bool, optional): Whether to preload the data into memory. Defaults to True.
    """
    annotations = Annotations(trigger_timing, [interval[-1]] * len(trigger_timing), trigger_labels)
    tmp_data = mne_data.copy()
    tmp_data.set_annotations(annotations)

    events_from_annot, _ = mne.events_from_annotations(tmp_data)

    return Epochs(
        mne_data,
        events_from_annot,
        tmin=interval[0],
        tmax=interval[-1],
        detrend=detrend,
        baseline=baseline,
        picks=channels,
        reject_by_annotation=reject_by_annotation,
        preload=preload)


def alphabetize(stimuli: List[str]) -> List[str]:
    """Alphabetize.

    Given a list of string stimuli, return a list of sorted stimuli by alphabet.
    """
    return sorted(stimuli, key=lambda x: re.sub(r'[^a-zA-Z0-9 \n\.]', 'ZZ', x).lower())


def inq_generator(query: List[str],
                  timing: List[float] = [1, 0.2],
                  color: List[str] = ['red', 'white'],
                  inquiry_count: int = 1,
                  stim_jitter: float = 0,
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

        times.append([timing[i] for i in range(len(timing) - 1)])
        base_timing = timing[-1]
        times[-1] += jittered_timing(base_timing, stim_jitter, stim_length)

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
        jitter: Optional[int] = None,
        color: List[str] = ['green', 'red', 'white'],
        stim_number: int = 10,
        stim_length: int = 10,
        stim_order: StimuliOrder = StimuliOrder.RANDOM,
        target_positions: TargetPositions = TargetPositions.RANDOM,
        nontarget_inquiries: int = 10,
        is_txt: bool = True) -> InquirySchedule:
    """Calibration Inquiry Generator.

    Generates inquiries with target letters in all possible positions.
        Args:
            alp(list[str]): stimuli
            timing(list[float]): Task specific timing for generator.
                [target, fixation, stimuli]
            jitter(int): jitter for stimuli timing. If None, no jitter is applied.
            color(list[str]): Task specific color for generator
                [target, fixation, stimuli]
            stim_number(int): number of trials in a inquiry
            stim_length(int): number of random stimuli to be created
            stim_order(StimuliOrder): ordering of stimuli in the inquiry
            target_positions(TargetPositions): positioning of targets to select for the inquiries
            nontarget_inquiries(int): percentage of inquiries for which target letter flashed is not in inquiry
            is_txt(bool): whether or not the stimuli type is text. False would be an image stimuli.
        Return:
            schedule_inq(tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)): scheduled inquiries
    """

    target_indexes = []
    no_target = None

    if (target_positions == target_positions.DISTRIBUTED):
        target_indexes = distributed_target_positions(stim_number, stim_length, nontarget_inquiries)
    else:
        # make list of random targets with correct number of non-target inquiries
        num_nontarget_inquiry = int((nontarget_inquiries / 100) * stim_number)
        num_target_inquiry = stim_number - num_nontarget_inquiry
        target_indexes = [no_target] * num_nontarget_inquiry
        target_indexes.extend(random.choices(range(stim_length), k=num_target_inquiry))
        random.shuffle(target_indexes)

    samples, times, colors = [], [], []

    for i in range(stim_number):
        inquiry = random.sample(alp, k=stim_length)

        if stim_order == StimuliOrder.ALPHABETICAL:
            inquiry = alphabetize(inquiry)

        target_index = target_indexes[i]
        if target_index is no_target:
            target = random.choice(list(set(alp) - set(inquiry)))
        else:
            target = inquiry[target_index]

        sample = [target, get_fixation(is_txt=is_txt), *inquiry]

        samples.append(sample)

        # timing for fixation and prompt
        init_timing = [timing[i] for i in range(len(timing) - 1)]
        # pull out timing for the inquiry stimuli
        stim_time = timing[-1]
        if jitter:
            _inq_timing = jittered_timing(stim_time, jitter, stim_length)
            inq_timing = init_timing + _inq_timing
        else:
            inq_timing = init_timing + [stim_time] * stim_length
        times.append(inq_timing)
        colors.append([color[i] for i in range(len(color) - 1)] +
                      [color[-1]] * stim_length)

    return InquirySchedule(samples, times, colors)


def jittered_timing(time: float, jitter: float, stim_count: int) -> List[float]:
    """Jittered timing.

    Using a base time and a jitter, generate a list (with length stim_count) of timing that is uniformly distributed.
    """
    assert time > jitter, (
        f'Jitter timing [{jitter}] must be less than stimuli timing =[{time}] in the inquiry.')
    return np.random.uniform(low=time - jitter, high=time + jitter, size=(stim_count,)).tolist()


def distributed_target_positions(stim_number: int, stim_length: int, nontarget_inquiries: int) -> list:
    """Distributed Target Positions.

    Generates evenly distributed target positions, including target letter not flashed at all, and shuffles them.
    Args:
        stim_number(int): Number of trials in calibration
        stim_length(int): Number of stimuli in each inquiry
        nontarget_inquiries(int): percentage of iquiries for which target letter flashed is not in inquiry

    Return distributed_target_positions(list): targets: array of target indexes to be chosen
    """

    targets = []
    no_target = None

    # find number of target and nontarget inquiries
    num_nontarget_inquiry = int(stim_number * (nontarget_inquiries / 100))
    num_target_inquiry = stim_number - num_nontarget_inquiry

    # find number each target position is repeated, and remaining number
    num_pos = (int)(num_target_inquiry / stim_length)
    num_rem_pos = (num_target_inquiry % stim_length)

    # add correct number of None's for nontarget inquiries
    targets = [no_target] * num_nontarget_inquiry

    # add distributed list of target positions
    targets.extend(list(range(stim_length)) * num_pos)

    # pick leftover positions randomly
    rem_pos = list(range(stim_length))
    random.shuffle(rem_pos)
    rem_pos = rem_pos[0:num_rem_pos]
    targets.extend(rem_pos)

    # shuffle targets
    random.shuffle(targets)

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
    # sd.play returns immediately (according to the docs); wait for the duration of the sound
    duration = len(data) / fs
    core.wait(duration)
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


def ssvep_to_code(refresh_rate: int = 60, flicker_rate: int = 10) -> List[int]:
    """Convert SSVEP to Code.

    Converts a SSVEP (steady state visual evoked potential; ex. 10 Hz) to a code (0,1)
    given the refresh rate of the monitor (Hz) provided and a desired flicker rate (Hz).

    Parameters:
    -----------
        refresh_rate: int, refresh rate of the monitor (Hz)
        flicker_rate: int, desired flicker rate (Hz)
    Returns:
    --------
        list of 0s and 1s that represent the code for the SSVEP on the monitor.
    """
    if flicker_rate > refresh_rate:
        raise BciPyCoreException('flicker rate cannot be greater than refresh rate')
    if flicker_rate <= 1:
        raise BciPyCoreException('flicker rate must be greater than 1')

    # get the number of frames per flicker
    length_flicker = refresh_rate / flicker_rate

    if length_flicker.is_integer():
        length_flicker = int(length_flicker)
    else:
        err_message = f'flicker rate={flicker_rate} is not an integer multiple of refresh rate={refresh_rate}'
        log.exception(err_message)
        raise BciPyCoreException(err_message)

    # start the first frames as off (0) for length of flicker;
    # it will then toggle on (1)/ off (0) for length of flicker until all frames are filled for refresh rate.
    t = 0
    codes = []
    for _ in range(flicker_rate):
        codes += [t] * length_flicker
        t = 1 - t

    return codes
