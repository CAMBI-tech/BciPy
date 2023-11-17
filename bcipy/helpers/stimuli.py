import glob
import itertools
import logging
import random
import re
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from os import path, sep
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

from pandas import Series
from PIL import Image

from bcipy.helpers.exceptions import BciPyCoreException
from bcipy.helpers.list import grouper

# Prevents pillow from filling the console with debug info
logging.getLogger('PIL').setLevel(logging.WARNING)

import mne
import numpy as np
import sounddevice as sd
import soundfile as sf
from mne import Annotations, Epochs
from mne.io import RawArray
from psychopy import core

log = logging.getLogger(__name__)
DEFAULT_FIXATION_PATH = 'bcipy/static/images/main/PLUS.png'
DEFAULT_TEXT_FIXATION = '+'
NO_TARGET_INDEX = None


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
                a label of [0, K-1] indicates the position of `target_label`, or label of [0 ... 0] indicates
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

                # If prestimulus buffer is used, we add it here so that trigger timings will
                # still line up with trial onset
                trial_triggers.append((trigger - first_trigger) + prestimulus_samples)
            reshaped_trigger_timing.append(trial_triggers)
            start = first_trigger - prestimulus_samples
            stop = first_trigger + num_samples_per_inq + buffer_samples
            reshaped_data.append(eeg_data[:, start:stop])

        return np.stack(reshaped_data, 1), labels, reshaped_trigger_timing

    @staticmethod
    def extract_trials(
            inquiries: np.ndarray,
            samples_per_trial: int,
            inquiry_timing: List[List[float]],
            prestimulus_samples: int = 0) -> np.ndarray:
        """Extract Trials.

        After using the InquiryReshaper, it may be necessary to further trial the data for processing.
        Using the number of samples and inquiry timing, the data is reshaped from Channels, Inquiry, Samples to
        Channels, Trials, Samples. These should match with the trials extracted from the TrialReshaper given the same
        slicing parameters.

        Parameters
        ----------
        inquiries : np.ndarray
            shape (Channels, Inquiries, Samples)
        samples_per_trial : int
            number of samples per trial
        inquiry_timing : List[List[float]]
            For each inquiry, a list of the sample index where each trial begins
        prestimulus_samples : int, optional
            Number of samples to move the start of each trial in each inquiry, by default 0.
            This is useful if wanting to use baseline intervals before the trial onset, along with the trial data.

        Returns
        -------
        np.ndarray
            shape (Channels, Trials, Samples)
        """
        new_trials = []
        num_inquiries = inquiries.shape[1]
        for inquiry_idx, timing in zip(range(num_inquiries), inquiry_timing):  # C x I x S

            # time == samples from the start of the inquiry
            for time in timing:
                start = time - prestimulus_samples
                end = time + samples_per_trial

                try:
                    new_trials.append(inquiries[:, inquiry_idx, start:end])
                except IndexError:  # pragma: no cover
                    raise BciPyCoreException(
                        f'InquiryReshaper.extract_trials: index out of bounds. \n'
                        f'Inquiry: [{inquiry_idx}] from {start}:{end}. init_time: {time}, '
                        f'prestimulus_samples: {prestimulus_samples}, samples_per_trial: {samples_per_trial} \n')
        return np.stack(new_trials, 1)  # C x T x S


class GazeReshaper:
    def __call__(self,
                 inq_start_times: List[float],
                 target_symbols: List[str],
                 gaze_data: np.ndarray,
                 sample_rate: int,
                 symbol_set: List[str],
                 channel_map: List[int] = None,
                 ) -> dict:
        """Extract inquiry data and labels. Different from the EEG inquiry, the gaze inquiry window starts with
        the first flicker and ends with the last flicker in the inquiry. Each inquiry has a length of ~3 seconds.
        The labels are provided in the target_symbols list. It returns a Dict, where keys are the target symbols and
        the values are inquiries (appended in order of appearance) where the corresponding target symbol is prompted.
        Optional outputs:
        reshape_data is the list of data reshaped into (Inquiries, Channels, Samples), where inquirires are appended
        in chronological order. labels returns the list of target symbols in each inquiry.

        Args:
            inq_start_times (List[float]): Timestamp of each event in seconds
            target_symbols (List[str]): Prompted symbol in each inquiry
            gaze_data (np.ndarray): shape (channels, samples) eye tracking data
            sample_rate (int): sample rate of data provided in eeg_data
            channel_map (List[int], optional): Describes which channels to include or discard.
                Defaults to None; all channels will be used.

        Returns:
            data_by_targets (dict): Dictionary where keys are the symbol set and values are the appended inquiries
            for each symbol. dict[Key] = (np.ndarray) of shape (Channels, Samples)

            reshaped_data (List[float]) [optional]: inquiry data of shape (Inquiries, Channels, Samples)
            labels (List[str]) [optional] : Target symbol in each inquiry.
        """
        if channel_map:
            # Remove the channels that we are not interested in
            channels_to_remove = [idx for idx, value in enumerate(channel_map) if value == 0]
            gaze_data = np.delete(gaze_data, channels_to_remove, axis=0)

        # Find the value closest to (& greater than) inq_start_times
        gaze_data_timing = gaze_data[-1, :].tolist()

        start_times = []
        for times in inq_start_times:
            temp = list(filter(lambda x: x > times, gaze_data_timing))
            if len(temp) > 0:
                start_times.append(temp[0])

        triggers = []
        for val in start_times:
            triggers.append(gaze_data_timing.index(val))

        # Label for every inquiry
        labels = target_symbols

        # Create a dictionary with symbols as keys and data as values
        # 'A': [], 'B': [] ...
        data_by_targets = {}
        for symbol in symbol_set:
            data_by_targets[symbol] = []

        window_length = 3  # seconds, total length of flickering after prompt for each inquiry

        reshaped_data = []
        # Merge the inquiries if they have the same target letter:
        for i, inquiry_index in enumerate(triggers):
            start = inquiry_index
            stop = int(inquiry_index + (sample_rate * window_length))   # (60 samples * 3 seconds)
            # Check if the data exists for the inquiry:
            if stop > len(gaze_data[0, :]):
                continue

            reshaped_data.append(gaze_data[:, start:stop])
            # (Optional) extracted data (Inquiries x Channels x Samples)

            # Populate the dict by appending the inquiry to the correct key:
            data_by_targets[labels[i]].append(gaze_data[:, start:stop])

        # After populating, flatten the arrays in the dictionary to (Channels x Samples):
        for symbol in symbol_set:
            if len(data_by_targets[symbol]) > 0:
                data_by_targets[symbol] = np.transpose(np.array(data_by_targets[symbol]), (1, 0, 2))
                data_by_targets[symbol] = np.reshape(data_by_targets[symbol], (len(data_by_targets[symbol]), -1))

            # Note that this is a workaround to the issue of having different number of targetness in
            # each symbol. If a target symbol is prompted more than once, the data is appended to the dict as a list.
            # Which is why we need to convert it to a (np.ndarray) and flatten the dimensions.
            # This is not ideal, but it works for now.

        # return np.stack(reshaped_data, 0), labels
        return data_by_targets

    @staticmethod
    def centralize_all_data(data, symbol_pos):
        """ Using the symbol locations in matrix, centralize all data (in Tobii units).
        This data will only be used in certain model types.
        Args:
            data (np.ndarray): Data in shape of num_channels x num_samples
            symbol_pos (np.ndarray(float)): Array of the current symbol posiiton in Tobii units
        Returns:
            data (np.ndarray): Centralized data in shape of num_channels x num_samples
        """
        for i in range(len(data)):
            data[i] = data[i] - symbol_pos
        return data


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

        Parameters
        ----------
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

        Returns
        -------
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


def update_inquiry_timing(timing: List[List[float]], downsample: int) -> List[List[float]]:
    """Update inquiry timing to reflect downsampling."""

    for i, inquiry in enumerate(timing):
        for j, time in enumerate(inquiry):
            timing[i][j] = time // downsample

    return timing


def mne_epochs(mne_data: RawArray,
               trigger_timing: List[float],
               trial_length: float,
               trigger_labels: List[int],
               baseline: Tuple[float, float] = (None, 0)) -> Epochs:
    """MNE Epochs.

    Using an MNE RawArray, reshape the data given trigger information. If two labels present [0, 1],
    each may be accessed by numbered order. Ex. first_class = epochs['1'], second_class = epochs['2']
    """
    annotations = Annotations(trigger_timing, [trial_length] * len(trigger_timing), trigger_labels)
    mne_data.set_annotations(annotations)
    events_from_annot, _ = mne.events_from_annotations(mne_data)
    return Epochs(mne_data, events_from_annot, tmax=trial_length, baseline=baseline)


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
    """Given the query set, prepares the stimuli, color and timing

    Parameters
    ----------
        query(list[str]): list of queries to be shown
        timing(list[float]): Task specific timing for generator
        color(list[str]): Task specific color for generator
            First element is the target, second element is the fixation
            Observe that [-1] element represents the trial information
    Return
    ------
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

    Given set of elements and a value function over the set, picks the len_query
        number of elements with the best value.

    Parameters
    ----------
        selection_elements(list[str]): the set of elements
        val(list[float]): values for the corresponding elements
        len_query(int): number of elements to be picked from the set
        always_included(list[str]): subset of elements that should always be
            included in the result. Defaults to None.
    Return
    ------
        best_selection(list[str]): elements from selection_elements with the best values
    """

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

    Generates RSVPKeyboard inquiry by picking n-most likely letters.

    Parameters
    ----------
        alp(list[str]): alphabet (can be arbitrary)
        session_stimuli(ndarray[float]): quantifier metric for query selection
            dim(session_stimuli) = card(alp)!
        timing(list[float]): Task specific timing for generator
        color(list[str]): Task specific color for generator
            First element is the target, second element is the fixation
            Observe that [-1] element represents the trial information
        inquiry_count(int): number of random stimuli to be created
        stim_per_inquiry(int): number of trials in a inquiry
        stim_order(StimuliOrder): ordering of stimuli in the inquiry
        inq_constants(list[str]): list of letters that should always be
            included in every inquiry. If provided, must be alp items.
    Return
    ------
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


def generate_calibration_inquiries(
        alp: List[str],
        timing: List[float] = None,
        jitter: Optional[int] = None,
        color: List[str] = None,
        inquiry_count: int = 100,
        stim_per_inquiry: int = 10,
        stim_order: StimuliOrder = StimuliOrder.RANDOM,
        target_positions: TargetPositions = TargetPositions.RANDOM,
        percentage_without_target: int = 0,
        is_txt: bool = True) -> InquirySchedule:
    """
    Generates inquiries with target letters in all possible positions.

    This function attempts to display all symbols as targets an equal number of
    times when stim_order is RANDOM. When the stim_order is ALPHABETICAL there
    is much more variation in the counts (target distribution takes priority)
    and some symbols may not appear as targets depending on the inquiry_count.
    The frequency that each symbol is displayed as a nontarget should follow a
    uniform distribution.

    Parameters
    ----------
        alp(list[str]): stimuli
        timing(list[float]): Task specific timing for generator.
            [target, fixation, stimuli]
        jitter(int): jitter for stimuli timing. If None, no jitter is applied.
        color(list[str]): Task specific color for generator
            [target, fixation, stimuli]
        inquiry_count(int): number of inquiries in a calibration
        stim_per_inquiry(int): number of stimuli in each inquiry
        stim_order(StimuliOrder): ordering of stimuli in the inquiry
        target_positions(TargetPositions): positioning of targets to select for the inquiries
        percentage_without_target(int): percentage of inquiries for which target letter flashed is not in inquiry
        is_txt(bool): whether the stimuli type is text. False would be an image stimuli.

    Return
    ------
        schedule_inq(tuple(
            samples[list[list[str]]]: list of inquiries
            timing(list[list[float]]): list of timings
            color(list(list[str])): list of colors)): scheduled inquiries
    """
    if timing is None:
        timing = [0.5, 1, 0.2]
    if color is None:
        color = ['green', 'red', 'white']
    assert len(
        timing
    ) == 3, "timing must include values for [target, fixation, stimuli]"
    time_target, time_fixation, time_stim = timing
    fixation = get_fixation(is_txt)

    target_indexes = generate_target_positions(inquiry_count, stim_per_inquiry,
                                               percentage_without_target,
                                               target_positions)
    if stim_order == StimuliOrder.ALPHABETICAL:
        targets = None
    else:
        targets = generate_targets(alp, inquiry_count,
                                   percentage_without_target)
    inquiries = generate_inquiries(alp, inquiry_count, stim_per_inquiry,
                                   stim_order)
    samples = []
    target = None
    for i in range(inquiry_count):
        inquiry = inquiries[i]
        target_pos = target_indexes[i]
        target = inquiry_target(inquiry,
                                target_pos,
                                symbols=alp,
                                next_targets=targets,
                                last_target=target)
        samples.append([target, fixation, *inquiry])

    times = [[
        time_target, time_fixation,
        *generate_inquiry_stim_timing(time_stim, stim_per_inquiry, jitter)
    ] for _ in range(inquiry_count)]

    inquiry_colors = color[0:2] + [color[-1]] * stim_per_inquiry
    colors = [inquiry_colors for _ in range(inquiry_count)]

    return InquirySchedule(samples, times, colors)


def inquiry_target_counts(inquiries: List[List[str]],
                          symbols: List[str]) -> dict:
    """Count the number of times each symbol was presented as a target.

    Args:
        inquiries - list of inquiries where each inquiry is structured as
            [target, fixation, *stim]
        symbols - list of all possible symbols
    """
    target_presentations = [inq[0] for inq in inquiries if inq[0] in inq[2:]]
    counter = dict.fromkeys(symbols, 0)
    for target in target_presentations:
        counter[target] += 1
    return counter


def inquiry_nontarget_counts(inquiries: List[List[str]],
                             symbols: List[str]) -> dict:
    """Count the number of times each symbol was presented as a nontarget."""
    counter = dict.fromkeys(symbols, 0)
    for inq in inquiries:
        target, _fixation, *stimuli = inq
        for stim in stimuli:
            if stim != target:
                counter[stim] += 1
    return counter


def inquiry_stats(inquiries: List[List[str]],
                  symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Descriptive stats for the number of times each target and nontarget
    symbol is shown in the inquiries"""

    target_stats = dict(
        Series(Counter(inquiry_target_counts(inquiries, symbols))).describe())

    nontarget_stats = dict(
        Series(Counter(inquiry_nontarget_counts(inquiries,
                                                symbols))).describe())

    return {
        'target_symbols': target_stats,
        'nontarget_symbols': nontarget_stats
    }


def generate_inquiries(symbols: List[str], inquiry_count: int,
                       stim_per_inquiry: int,
                       stim_order: StimuliOrder) -> List[List[str]]:
    """Generate a list of inquiries. For each inquiry no symbols are repeated.
    Inquiries do not include the target or fixation. Symbols should be
    distributed uniformly across inquiries.

    Args:

        symbols - values from which to select
        inquiry_count - total number of inquiries to generate
        stim_per_inquiry - length of each inquiry
        stim_order - ordering of results
    """
    return [
        generate_inquiry(symbols=symbols,
                         length=stim_per_inquiry,
                         stim_order=stim_order) for _ in range(inquiry_count)
    ]


def generate_inquiry(symbols: List[str], length: int,
                     stim_order: StimuliOrder) -> List[str]:
    """Generate an inquiry from the list of symbols. No symbols are repeated
    in the output list. Output does not include the target or fixation.

    Args:
        symbols - values from which to select
        length - number of items in the return list
        stim_order - ordering of results
    """
    inquiry = random.sample(symbols, k=length)
    if stim_order == StimuliOrder.ALPHABETICAL:
        inquiry = alphabetize(inquiry)
    return inquiry


def inquiry_target(inquiry: List[str],
                   target_position: Optional[int],
                   symbols: List[str],
                   next_targets: List[str] = None,
                   last_target: str = None) -> str:
    """Returns the target for the given inquiry. If the optional
    target position is not provided a target will randomly be selected from
    the list of symbols and will not be in the inquiry.

    Args:
        inquiry - list of symbols to be presented
        target_position - optional position within the list of the target sym
        symbols - used if position is not provided to select a random symbol
            as the target.
        next_targets - list of targets from which to select
        last_target - target from the previous inquiry; used to avoid selecting
            the same target consecutively.

    Returns target symbol
    """
    if target_position is None:
        return random.choice(list(set(symbols) - set(inquiry)))

    if next_targets:
        # select the next target from the symbols in the current inquiry.
        # if none of the symbols in the current inquiry are in the choices, the
        # target is determined strictly from the target_position.
        choice_index = None
        symbol_inquiry_position = None
        target = None
        for position, symbol in enumerate(inquiry):
            # avoid repeating targets
            if symbol == last_target:
                continue
            try:
                symbol_index = next_targets.index(symbol)
                if choice_index is None or symbol_index < choice_index:
                    choice_index = symbol_index
                    symbol_inquiry_position = position
                    target = symbol
            except ValueError:
                continue

        if target is not None:
            # update next_targets list
            next_targets.remove(target)  # removes first occurrence of value

            # update inquiry to set the target at the expected position.
            symbol_at_target_position = inquiry[target_position]
            inquiry[symbol_inquiry_position] = symbol_at_target_position
            inquiry[target_position] = target

    return inquiry[target_position]


def generate_inquiry_stim_timing(time_stim: float, length: int,
                                 jitter: bool) -> List[float]:
    """Generate stimuli timing values for a given inquiry.

    Args:
        time_stim: seconds to display each stimuli
        length: Number of timings to generate
        jitter: whether the timing should be jittered.

    Returns list of times (in seconds)
    """
    if jitter:
        return jittered_timing(time_stim, jitter, length)

    return [time_stim] * length


def jittered_timing(time: float, jitter: float,
                    stim_count: int) -> List[float]:
    """Jittered timing.

    Using a base time and a jitter, generate a list (with length stim_count) of
    timing that is uniformly distributed.
    """
    assert time > jitter, (
        f'Jitter timing [{jitter}] must be less than stimuli timing =[{time}] in the inquiry.'
    )
    return np.random.uniform(low=time - jitter,
                             high=time + jitter,
                             size=(stim_count, )).tolist()


def compute_counts(inquiry_count: int,
                   percentage_without_target: int) -> Tuple[int, int]:
    """Determine the number of inquiries that should display targets and the
    number that should not.

    Args:
        inquiry_count: Number of inquiries in calibration
        percentage_without_target: percentage of inquiries for which
            target letter flashed is not in inquiry

    Returns tuple of (target_count, no_target_count)
    """
    no_target_count = int(inquiry_count * (percentage_without_target / 100))
    target_count = inquiry_count - no_target_count
    return target_count, no_target_count


def generate_target_positions(inquiry_count: int, stim_per_inquiry: int,
                              percentage_without_target: int,
                              distribution: TargetPositions) -> List[int]:
    """
    Generates target positions distributed according to the provided parameter.

    Args:
        inquiry_count: Number of inquiries in calibration
        stim_per_inquiry: Number of stimuli in each inquiry
        percentage_without_target: percentage of inquiries for which
            target letter flashed is not in inquiry
        distribution: specifies how targets should be distributed

    Returns list of indexes
    """
    if distribution is TargetPositions.DISTRIBUTED:
        return distributed_target_positions(inquiry_count, stim_per_inquiry,
                                            percentage_without_target)
    return random_target_positions(inquiry_count, stim_per_inquiry,
                                   percentage_without_target)


def distributed_target_positions(inquiry_count: int, stim_per_inquiry: int,
                                 percentage_without_target: int) -> list:
    """Distributed Target Positions.

    Generates evenly distributed target positions, including target letter
    not flashed at all, and shuffles them.

    Args:
        inquiry_count(int): Number of inquiries in calibration
        stim_per_inquiry(int): Number of stimuli in each inquiry
        percentage_without_target(int): percentage of inquiries for which
            target letter flashed is not in inquiry

    Return distributed_target_positions(list): targets: array of target
    indexes to be chosen
    """

    targets = []

    # find number of target and no_target inquiries
    target_count, no_target_count = compute_counts(inquiry_count,
                                                   percentage_without_target)

    # find number each target position is repeated, and remaining number
    num_pos = (int)(target_count / stim_per_inquiry)
    num_rem_pos = (target_count % stim_per_inquiry)

    # add correct number of None's for nontarget inquiries
    targets = [NO_TARGET_INDEX] * no_target_count

    # add distributed list of target positions
    targets.extend(list(range(stim_per_inquiry)) * num_pos)

    # pick leftover positions randomly
    rem_pos = list(range(stim_per_inquiry))
    random.shuffle(rem_pos)
    rem_pos = rem_pos[0:num_rem_pos]
    targets.extend(rem_pos)

    # shuffle targets
    random.shuffle(targets)

    return targets


def random_target_positions(inquiry_count: int, stim_per_inquiry: int,
                            percentage_without_target: int) -> list:
    """Generates randomly distributed target positions, including target letter
    not flashed at all, and shuffles them.

    Args:
        inquiry_count(int): Number of inquiries in calibration
        stim_per_inquiry(int): Number of stimuli in each inquiry
        percentage_without_target(int): percentage of inquiries for which
            target letter flashed is not in inquiry

    Return list of target indexes to be chosen
    """
    target_count, no_target_count = compute_counts(inquiry_count,
                                                   percentage_without_target)

    target_indexes = [NO_TARGET_INDEX] * no_target_count
    target_indexes.extend(
        random.choices(range(stim_per_inquiry), k=target_count))
    random.shuffle(target_indexes)
    return target_indexes


def generate_targets(symbols: List[str], inquiry_count: int,
                     percentage_without_target: int) -> List[str]:
    """Generates list of target symbols. Generates an equal number of each
    target. The resulting list may be less than the inquiry_count. Used for
    sampling without replacement to get approximately equal numbers of each
    target.

    Args:
        symbols:
        inquiry_count: number of inquiries in calibration
        percentage_without_target: percentage of inquiries for which
            target letter flashed is not in inquiry
    """
    target_count, no_target_count = compute_counts(inquiry_count,
                                                   percentage_without_target)

    # each symbol should appear at least once
    symbol_count = int(target_count / len(symbols)) or 1
    targets = symbols * symbol_count
    random.shuffle(targets)
    return targets


def target_index(inquiry: List[str]) -> int:
    """Given an inquiry, return the index of the target within the choices and
    None if the target is not included as a choice.

    Parameters
    ----------
        inquiry - list of [target, fixation, *choices]

    >>> inquiry = ['T', '+', 'G', 'J', 'K', 'L', 'M', 'Q', 'T', 'V', 'X', '<']
    >>> target_index(inquiry)
    6
    >>> inquiry = ['A', '+', 'G', 'J', 'K', 'L', 'M', 'Q', 'T', 'V', 'X', '<']
    >>> target_index(inquiry)
    None
    """
    assert len(inquiry) > 3, "Not enough choices"
    target, _fixation, *choices = inquiry
    try:
        return choices.index(target)
    except ValueError:
        return None


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
        return DEFAULT_TEXT_FIXATION
    else:
        return DEFAULT_FIXATION_PATH
