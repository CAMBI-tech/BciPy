# mypy: disable-error-code="arg-type"
import glob
import itertools
import logging
import random
import re
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from os import path, sep
from typing import (Any, Dict, Iterator, List, NamedTuple, Optional, Tuple,
                    Union)

import mne
import numpy as np
import sounddevice as sd
import soundfile as sf
from mne import Annotations, Epochs
from mne.io import RawArray
from pandas import Series
from PIL import Image
from psychopy import core

from bcipy.config import (DEFAULT_FIXATION_PATH, DEFAULT_TEXT_FIXATION,
                          SESSION_LOG_FILENAME)
from bcipy.core.list import grouper
from bcipy.core.symbols import alphabet
from bcipy.exceptions import BciPyCoreException

# Prevents pillow from filling the console with debug info
logging.getLogger('PIL').setLevel(logging.WARNING)
log = logging.getLogger(SESSION_LOG_FILENAME)

NO_TARGET_INDEX = None


class StimuliOrder(Enum):
    """Defines the ordering of stimuli for inquiry.

    Attributes:
        RANDOM (str): Random ordering of stimuli.
        ALPHABETICAL (str): Alphabetical ordering of stimuli.
    """
    RANDOM = 'random'
    ALPHABETICAL = 'alphabetical'

    @classmethod
    def list(cls) -> list:
        """Returns all enum values as a list.

        Returns:
            list: List of all enum values.
        """
        return list(map(lambda c: c.value, cls))


class TargetPositions(Enum):
    """Defines the positions of targets within the inquiry.

    Attributes:
        RANDOM (str): Random positioning of targets.
        DISTRIBUTED (str): Evenly distributed positioning of targets.
    """
    RANDOM = 'random'
    DISTRIBUTED = 'distributed'

    @classmethod
    def list(cls) -> list:
        """Returns all enum values as a list.

        Returns:
            list: List of all enum values.
        """
        return list(map(lambda c: c.value, cls))


class PhotoDiodeStimuli(Enum):
    """Defines unicode stimuli needed for testing system timing.

    Attributes:
        EMPTY (str): Box with a white border, no fill (□).
        SOLID (str): Solid white box (■).
    """
    EMPTY = '\u25A1'  # box with a white border, no fill
    SOLID = '\u25A0'  # solid white box

    @classmethod
    def list(cls) -> list:
        """Returns all enum values as a list.

        Returns:
            list: List of all enum values.
        """
        return list(map(lambda c: c.value, cls))


class InquirySchedule(NamedTuple):
    """Schedule for the next inquiries to present, where each inquiry specifies
    the stimulus, duration, and color information.

    Attributes:
        stimuli (List[Any]): List of stimuli for each inquiry.
        durations (Union[List[List[float]], List[float]]): Duration for each stimulus.
        colors (Union[List[List[str]], List[str]]): Color for each stimulus.
    """
    stimuli: List[Any]
    durations: Union[List[List[float]], List[float]]
    colors: Union[List[List[str]], List[str]]

    def inquiries(self) -> Iterator[Tuple]:
        """Generator that iterates through each Inquiry.

        Yields:
            Tuple: Tuple of (stim, duration, color) for each inquiry.
        """
        count = len(self.stimuli)
        index = 0
        while index < count:
            yield (self.stimuli[index], self.durations[index],
                   self.colors[index])
            index += 1


class Reshaper(ABC):
    """Abstract base class for reshaping data in BCI experiments."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Reshape data for a specific paradigm.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Reshaped data.
        """
        ...


class InquiryReshaper:
    """Reshapes EEG data, timing, and labels for inquiries in BCI experiments."""

    def __call__(self,
                 trial_targetness_label: List[str],
                 timing_info: List[float],
                 eeg_data: np.ndarray,
                 sample_rate: int,
                 trials_per_inquiry: int,
                 offset: float = 0,
                 channel_map: Optional[List[int]] = None,
                 poststimulus_length: float = 0.5,
                 prestimulus_length: float = 0.0,
                 transformation_buffer: float = 0.0,
                 target_label: str = 'target') -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
        """Extract inquiry data and labels.

        Args:
            trial_targetness_label (List[str]): Labels each trial as "target", "non-target", etc.
            timing_info (List[float]): Timestamp of each event in seconds.
            eeg_data (np.ndarray): Shape (channels, samples) preprocessed EEG data.
            sample_rate (int): Sample rate of data provided in eeg_data.
            trials_per_inquiry (int): Number of trials in each inquiry.
            offset (float, optional): Any calculated or hypothesized offsets in timings. Defaults to 0.
            channel_map (List[int], optional): Describes which channels to include or discard. Defaults to None.
            poststimulus_length (float, optional): Time in seconds needed after the last trial. Defaults to 0.5.
            prestimulus_length (float, optional): Time in seconds needed before the first trial. Defaults to 0.0.
            transformation_buffer (float, optional): Time in seconds to buffer the end of the inquiry. Defaults to 0.0.
            target_label (str, optional): Label of target symbol. Defaults to "target".

        Returns:
            Tuple[np.ndarray, np.ndarray, List[List[float]]]:
                - reshaped_data: Inquiry data of shape (Channels, Inquiries, Samples)
                - labels: Integer label for each inquiry
                - reshaped_trigger_timing: For each inquiry, a list of the sample index where each trial begins
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
            (n_inquiry, trials_per_inquiry), dtype=np.longlong
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
            inquiry_timing: List[List[int]],
            prestimulus_samples: int = 0) -> np.ndarray:
        """Extract trials from inquiry data.

        Args:
            inquiries (np.ndarray): Shape (Channels, Inquiries, Samples).
            samples_per_trial (int): Number of samples per trial.
            inquiry_timing (List[List[int]]): For each inquiry, a list of the sample index where each trial begins.
            prestimulus_samples (int, optional): Number of samples to move the start of each trial. Defaults to 0.

        Returns:
            np.ndarray: Shape (Channels, Trials, Samples).

        Raises:
            BciPyCoreException: If index is out of bounds when extracting trials.
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
    """Reshapes gaze trajectory data and labels for inquiries in BCI experiments."""

    def __call__(self,
                 inq_start_times: List[float],
                 target_symbols: List[str],
                 gaze_data: np.ndarray,
                 sample_rate: int,
                 stimulus_duration: float,
                 num_stimuli_per_inquiry: int,
                 symbol_set: List[str] = alphabet(),
                 channel_map: Optional[List[int]] = None,
                 ) -> Tuple[dict, list, List[str]]:
        """Extract gaze trajectory data and labels.

        Args:
            inq_start_times (List[float]): Timestamp of each event in seconds.
            target_symbols (List[str]): Prompted symbol in each inquiry.
            gaze_data (np.ndarray): Shape (channels, samples) eye tracking data.
            sample_rate (int): Sample rate of eye tracker data.
            stimulus_duration (float): Duration of flash time (in seconds) for each trial.
            num_stimuli_per_inquiry (int): Number of stimuli in each inquiry.
            symbol_set (List[str], optional): List of all symbols for the task. Defaults to alphabet().
            channel_map (List[int], optional): Describes which channels to include or discard. Defaults to None.

        Returns:
            Tuple[dict, list, List[str]]:
                - data_by_targets: Dictionary where keys are the symbol set, and values are the appended inquiries for each symbol.
                - reshaped_data: Inquiry data of shape (Inquiries, Channels, Samples).
                - labels: Target symbol in each inquiry.
        """
        # Find the timestamp value closest to (& greater than) inq_start_times.
        # Lsl timestamps are the last row in the gaze_data
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
        data_by_targets_dict: Dict[str, list] = {}
        for symbol in symbol_set:
            data_by_targets_dict[symbol] = []

        buffer = 0.5  # seconds, buffer for each inquiry
        # NOTE: This buffer is used to account for the screen downtime between each stimulus.
        # A better way of handling this buffer would be subtracting the flash time of the
        # second symbol from the first symbol, which gives a more accurate representation of
        # "stimulus duration".
        window_length = (stimulus_duration + buffer) * num_stimuli_per_inquiry   # in seconds

        reshaped_data = []
        # Merge the inquiries if they have the same target letter:
        for i, inquiry_index in enumerate(triggers):
            start = inquiry_index
            stop = int(inquiry_index + (sample_rate * window_length))
            # Check if the data exists for the inquiry:
            if stop > len(gaze_data[0, :]):
                continue
            # (Optional) extracted data (Inquiries x Channels x Samples)
            reshaped_data.append(gaze_data[:, start:stop])

            # Populate the dict by appending the inquiry to the corresponding key:
            data_by_targets_dict[labels[i]].append(gaze_data[:, start:stop])

        return data_by_targets_dict, reshaped_data, labels

    def centralize_all_data(self, data: np.ndarray, symbol_pos: np.ndarray) -> np.ndarray:
        """Centralize all data using symbol locations in matrix (Tobii units).

        Args:
            data (np.ndarray): Data in shape of num_samples x num_dimensions.
            symbol_pos (np.ndarray): Array of the current symbol position in Tobii units.

        Returns:
            np.ndarray: Centralized data in shape of num_samples x num_dimensions.
        """
        new_data = np.copy(data)
        for i in range(len(data)):
            new_data[i] = data[i] - symbol_pos

        return new_data


class TrialReshaper(Reshaper):
    """Reshapes EEG data, timing, and labels for individual trials in BCI experiments."""

    def __call__(self,
                 trial_targetness_label: list,
                 timing_info: list,
                 eeg_data: np.ndarray,
                 sample_rate: int,
                 offset: float = 0,
                 channel_map: Optional[List[int]] = None,
                 poststimulus_length: float = 0.5,
                 prestimulus_length: float = 0.0,
                 target_label: str = "target") -> Tuple[np.ndarray, np.ndarray]:
        """Extract trial data and labels.

        Args:
            trial_targetness_label (list): Labels each trial as "target", "non-target", etc.
            timing_info (list): Timestamp of each event in seconds.
            eeg_data (np.ndarray): Shape (channels, samples) preprocessed EEG data.
            sample_rate (int): Sample rate of preprocessed EEG data.
            offset (float, optional): Any calculated or hypothesized offsets in timings. Defaults to 0.
            channel_map (List, optional): Describes which channels to include or discard. Defaults to None.
            poststimulus_length (float, optional): Time in seconds needed after the last trial. Defaults to 0.5.
            prestimulus_length (float, optional): Time in seconds needed before the first trial. Defaults to 0.0.
            target_label (str, optional): Label of target symbol. Defaults to "target".

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - trial_data: Reshaped data of shape (channels, trials, samples)
                - labels: Integer label for each trial
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
        targetness_labels = np.zeros(len(triggers), dtype=np.longlong)
        reshaped_trials = []
        for trial_idx, (trial_label, trigger) in enumerate(zip(trial_targetness_label, triggers)):
            if trial_label == target_label:
                targetness_labels[trial_idx] = 1

            # For every channel append filtered channel data to trials
            reshaped_trials.append(eeg_data[:, trigger - prestim_samples: trigger + poststim_samples])

        return np.stack(reshaped_trials, 1), targetness_labels


def update_inquiry_timing(timing: List[List[int]], downsample: int) -> List[List[int]]:
    """Update inquiry timing to reflect downsampling.

    Args:
        timing (List[List[int]]): Original timing values for each inquiry.
        downsample (int): Downsampling factor.

    Returns:
        List[List[int]]: Updated timing values for each inquiry.
    """
    for i, inquiry in enumerate(timing):
        for j, time in enumerate(inquiry):
            timing[i][j] = int(time // downsample)

    return timing


def mne_epochs(mne_data: RawArray,
               trial_length: float,
               trigger_timing: Optional[List[float]] = None,
               trigger_labels: Optional[List[int]] = None,
               baseline: Optional[Tuple[Any, float]] = None,
               reject_by_annotation: bool = False,
               preload: bool = False) -> Epochs:
    """Create MNE Epochs from a RawArray and trigger information.

    Args:
        mne_data (RawArray): MNE RawArray object.
        trial_length (float): Length of each trial in seconds.
        trigger_timing (Optional[List[float]], optional): List of trigger times. Defaults to None.
        trigger_labels (Optional[List[int]], optional): List of trigger labels. Defaults to None.
        baseline (Optional[Tuple[Any, float]], optional): Baseline interval. Defaults to None.
        reject_by_annotation (bool, optional): Whether to reject epochs by annotation. Defaults to False.
        preload (bool, optional): Whether to preload the data. Defaults to False.

    Returns:
        Epochs: MNE Epochs object.
    """
    old_annotations = mne_data.annotations
    if trigger_timing and trigger_labels:
        new_annotations = Annotations(trigger_timing, [trial_length] * len(trigger_timing), trigger_labels)
        all_annotations = new_annotations + old_annotations
    else:
        all_annotations = old_annotations

    tmp_data = mne_data.copy()
    tmp_data.set_annotations(all_annotations)

    events_from_annot, _ = mne.events_from_annotations(tmp_data)

    if baseline is None:
        baseline = (0, 0)
        tmin = -0.1
    else:
        tmin = baseline[0]

    return Epochs(
        mne_data,
        events_from_annot,
        baseline=baseline,
        tmax=trial_length,
        tmin=tmin,
        proj=False,  # apply SSP projection to data. Defaults to True in Epochs.
        reject_by_annotation=reject_by_annotation,
        preload=preload)


def alphabetize(stimuli: List[str]) -> List[str]:
    """Return a list of sorted stimuli by alphabet.

    Args:
        stimuli (List[str]): List of string stimuli.

    Returns:
        List[str]: Alphabetically sorted list of stimuli.
    """
    return sorted(stimuli, key=lambda x: re.sub(r'[^a-zA-Z0-9 \n\.]', 'ZZ', x).lower())


def inq_generator(query: List[str],
                  timing: List[float] = [1, 0.2],
                  color: List[str] = ['red', 'white'],
                  inquiry_count: int = 1,
                  stim_jitter: float = 0,
                  stim_order: StimuliOrder = StimuliOrder.RANDOM,
                  is_txt: bool = True) -> InquirySchedule:
    """Prepare the stimuli, color, and timing for a set of inquiries.

    Args:
        query (List[str]): List of queries to be shown.
        timing (List[float], optional): Task specific timing for generator. Defaults to [1, 0.2].
        color (List[str], optional): Task specific color for generator. Defaults to ['red', 'white'].
        inquiry_count (int, optional): Number of inquiries to generate. Defaults to 1.
        stim_jitter (float, optional): Jitter to apply to stimulus timing. Defaults to 0.
        stim_order (StimuliOrder, optional): Ordering of stimuli. Defaults to StimuliOrder.RANDOM.
        is_txt (bool, optional): Whether the stimuli are text. Defaults to True.

    Returns:
        InquirySchedule: Scheduled inquiries with samples, timing, and color.
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
                   always_included: Optional[List[str]] = None) -> list:
    """Pick the len_query number of elements with the best value.

    Args:
        selection_elements (list): The set of elements.
        val (list): Values for the corresponding elements.
        len_query (int): Number of elements to be picked from the set.
        always_included (Optional[List[str]], optional): Subset of elements that should always be included. Defaults to None.

    Returns:
        list: Elements from selection_elements with the best values.
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
                           inq_constants: Optional[List[str]] = None) -> InquirySchedule:
    """Generate RSVPKeyboard inquiry by picking n-most likely letters.

    Args:
        alp (list): Alphabet (can be arbitrary).
        session_stimuli (np.ndarray): Quantifier metric for query selection.
        timing (List[float], optional): Task specific timing for generator. Defaults to [1, 0.2].
        color (List[str], optional): Task specific color for generator. Defaults to ['red', 'white'].
        stim_number (int, optional): Number of random stimuli to be created. Defaults to 1.
        stim_length (int, optional): Number of trials in an inquiry. Defaults to 10.
        stim_order (StimuliOrder, optional): Ordering of stimuli. Defaults to StimuliOrder.RANDOM.
        is_txt (bool, optional): Whether the stimuli are text. Defaults to True.
        inq_constants (Optional[List[str]], optional): Letters that should always be included. Defaults to None.

    Returns:
        InquirySchedule: Scheduled inquiries with samples, timing, and color.
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
        timing: Optional[List[float]] = None,
        jitter: Optional[int] = None,
        color: Optional[List[str]] = None,
        inquiry_count: int = 100,
        stim_per_inquiry: int = 10,
        stim_order: StimuliOrder = StimuliOrder.RANDOM,
        target_positions: TargetPositions = TargetPositions.RANDOM,
        percentage_without_target: int = 0,
        is_txt: bool = True) -> InquirySchedule:
    """Generate inquiries with target letters in all possible positions.

    Args:
        alp (List[str]): Stimuli.
        timing (Optional[List[float]], optional): Task specific timing for generator. Defaults to None.
        jitter (Optional[int], optional): Jitter for stimuli timing. Defaults to None.
        color (Optional[List[str]], optional): Task specific color for generator. Defaults to None.
        inquiry_count (int, optional): Number of inquiries in a calibration. Defaults to 100.
        stim_per_inquiry (int, optional): Number of stimuli in each inquiry. Defaults to 10.
        stim_order (StimuliOrder, optional): Ordering of stimuli. Defaults to StimuliOrder.RANDOM.
        target_positions (TargetPositions, optional): Positioning of targets. Defaults to TargetPositions.RANDOM.
        percentage_without_target (int, optional): Percentage of inquiries without a target. Defaults to 0.
        is_txt (bool, optional): Whether the stimuli type is text. Defaults to True.

    Returns:
        InquirySchedule: Scheduled inquiries with samples, timing, and color.
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
        inquiries (List[List[str]]): List of inquiries where each inquiry is structured as [target, fixation, *stim].
        symbols (List[str]): List of all possible symbols.

    Returns:
        dict: Dictionary mapping each symbol to its target count.
    """
    target_presentations = [inq[0] for inq in inquiries if inq[0] in inq[2:]]
    counter = dict.fromkeys(symbols, 0)
    for target in target_presentations:
        counter[target] += 1
    return counter


def inquiry_nontarget_counts(inquiries: List[List[str]],
                             symbols: List[str]) -> dict:
    """Count the number of times each symbol was presented as a nontarget.

    Args:
        inquiries (List[List[str]]): List of inquiries.
        symbols (List[str]): List of all possible symbols.

    Returns:
        dict: Dictionary mapping each symbol to its nontarget count.
    """
    counter = dict.fromkeys(symbols, 0)
    for inq in inquiries:
        target, _fixation, *stimuli = inq
        for stim in stimuli:
            if stim != target:
                counter[stim] += 1
    return counter


def inquiry_stats(inquiries: List[List[str]],
                  symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Descriptive stats for the number of times each target and nontarget symbol is shown in the inquiries.

    Args:
        inquiries (List[List[str]]): List of inquiries.
        symbols (List[str]): List of all possible symbols.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with stats for target and nontarget symbols.
    """

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

    Args:
        symbols (List[str]): Values from which to select.
        inquiry_count (int): Total number of inquiries to generate.
        stim_per_inquiry (int): Length of each inquiry.
        stim_order (StimuliOrder): Ordering of results.

    Returns:
        List[List[str]]: List of generated inquiries.
    """
    return [
        generate_inquiry(symbols=symbols,
                         length=stim_per_inquiry,
                         stim_order=stim_order) for _ in range(inquiry_count)
    ]


def generate_inquiry(symbols: List[str], length: int,
                     stim_order: StimuliOrder) -> List[str]:
    """Generate an inquiry from the list of symbols. No symbols are repeated in the output list.

    Args:
        symbols (List[str]): Values from which to select.
        length (int): Number of items in the return list.
        stim_order (StimuliOrder): Ordering of results.

    Returns:
        List[str]: Generated inquiry.
    """
    inquiry = random.sample(symbols, k=length)
    if stim_order == StimuliOrder.ALPHABETICAL:
        inquiry = alphabetize(inquiry)
    return inquiry


def inquiry_target(inquiry: List[str],
                   target_position: Optional[int],
                   symbols: List[str],
                   next_targets: Optional[List[str]] = None,
                   last_target: Optional[str] = None) -> str:
    """Returns the target for the given inquiry.

    Args:
        inquiry (List[str]): List of symbols to be presented.
        target_position (Optional[int]): Optional position within the list of the target symbol.
        symbols (List[str]): Used if position is not provided to select a random symbol as the target.
        next_targets (Optional[List[str]], optional): List of targets from which to select. Defaults to None.
        last_target (Optional[str], optional): Target from the previous inquiry. Defaults to None.

    Returns:
        str: Target symbol.
    """
    if target_position is None:
        return random.choice(list(set(symbols) - set(inquiry)))

    if next_targets:
        # select the next target from the symbols in the current inquiry.
        # if none of the symbols in the current inquiry are in the choices, the
        # target is determined strictly from the target_position.
        choice_index = None
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
        time_stim (float): Seconds to display each stimulus.
        length (int): Number of timings to generate.
        jitter (bool): Whether the timing should be jittered.

    Returns:
        List[float]: List of times (in seconds).
    """
    if jitter:
        return jittered_timing(time_stim, jitter, length)

    return [time_stim] * length


def jittered_timing(time: float, jitter: float,
                    stim_count: int) -> List[float]:
    """Generate a list of jittered timing values for stimuli.

    Args:
        time (float): Base time for each stimulus.
        jitter (float): Jitter to apply.
        stim_count (int): Number of stimuli.

    Returns:
        List[float]: List of jittered timing values.
    """
    assert time > jitter, (
        f'Jitter timing [{jitter}] must be less than stimuli timing =[{time}] in the inquiry.'
    )
    return np.random.uniform(low=time - jitter,
                             high=time + jitter,
                             size=(stim_count, )).tolist()


def compute_counts(inquiry_count: int,
                   percentage_without_target: int) -> Tuple[int, int]:
    """Determine the number of inquiries that should display targets and the number that should not.

    Args:
        inquiry_count (int): Number of inquiries in calibration.
        percentage_without_target (int): Percentage of inquiries without a target.

    Returns:
        Tuple[int, int]: Tuple of (target_count, no_target_count).
    """
    no_target_count = int(inquiry_count * (percentage_without_target / 100))
    target_count = inquiry_count - no_target_count
    return target_count, no_target_count


def generate_target_positions(inquiry_count: int, stim_per_inquiry: int,
                              percentage_without_target: int,
                              distribution: TargetPositions) -> List[int]:
    """Generate target positions distributed according to the provided parameter.

    Args:
        inquiry_count (int): Number of inquiries in calibration.
        stim_per_inquiry (int): Number of stimuli in each inquiry.
        percentage_without_target (int): Percentage of inquiries without a target.
        distribution (TargetPositions): Specifies how targets should be distributed.

    Returns:
        List[int]: List of indexes for target positions.
    """
    if distribution is TargetPositions.DISTRIBUTED:
        return distributed_target_positions(inquiry_count, stim_per_inquiry,
                                            percentage_without_target)
    return random_target_positions(inquiry_count, stim_per_inquiry,
                                   percentage_without_target)


def distributed_target_positions(inquiry_count: int, stim_per_inquiry: int,
                                 percentage_without_target: int) -> list:
    """Generate evenly distributed target positions, including target letter not flashed at all, and shuffle them.

    Args:
        inquiry_count (int): Number of inquiries in calibration.
        stim_per_inquiry (int): Number of stimuli in each inquiry.
        percentage_without_target (int): Percentage of inquiries without a target.

    Returns:
        list: Targets array of target indexes to be chosen.
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
    """Generate randomly distributed target positions, including target letter not flashed at all, and shuffle them.

    Args:
        inquiry_count (int): Number of inquiries in calibration.
        stim_per_inquiry (int): Number of stimuli in each inquiry.
        percentage_without_target (int): Percentage of inquiries without a target.

    Returns:
        list: List of target indexes to be chosen.
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
    """Generate a list of target symbols for calibration inquiries.

    Args:
        symbols (List[str]): List of possible symbols.
        inquiry_count (int): Number of inquiries in calibration.
        percentage_without_target (int): Percentage of inquiries without a target.

    Returns:
        List[str]: List of target symbols.
    """
    target_count, no_target_count = compute_counts(inquiry_count,
                                                   percentage_without_target)

    # each symbol should appear at least once
    symbol_count = int(target_count / len(symbols)) or 1
    targets = symbols * symbol_count
    random.shuffle(targets)
    return targets


def target_index(inquiry: List[str]) -> Optional[int]:
    """Return the index of the target within the choices, or None if not present.

    Args:
        inquiry (List[str]): List of [target, fixation, *choices].

    Returns:
        Optional[int]: Index of the target in choices, or None if not present.
    """
    assert len(inquiry) > 3, "Not enough choices"
    target, _fixation, *choices = inquiry
    try:
        return choices.index(target)
    except ValueError:
        return None


def get_task_info(experiment_length: int, task_color: str) -> Tuple[List[str], List[str]]:
    """Generate fixed RSVPKeyboard task text and color information for display.

    Args:
        experiment_length (int): Number of inquiries for the experiment.
        task_color (str): Task information display color.

    Returns:
        Tuple[List[str], List[str]]: Tuple of task text and color arrays.
    """

    # Do list comprehensions to get the arrays for the task we need.
    task_text_list = ['%s/%s' % (stim + 1, experiment_length)
                      for stim in range(experiment_length)]
    task_color_list = [str(task_color) for _ in range(experiment_length)]

    return (task_text_list, task_color_list)


def resize_image(image_path: str, screen_size: tuple, sti_height: float) -> Tuple[float, float]:
    """Return the width and height that a given image should be displayed at given the screen size and stimuli height.

    Args:
        image_path (str): Path to the image file.
        screen_size (tuple): Screen size as (width, height).
        sti_height (float): Desired stimuli height.

    Returns:
        Tuple[float, float]: Width and height for displaying the image.
    """
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
               trigger_name: Optional[str] = None,
               timing: list = []) -> list:
    """Play a sound file and optionally track timing and triggers.

    Args:
        sound_file_path (str): Path to the sound file.
        dtype (str, optional): Type of sound (e.g., 'float32'). Defaults to 'float32'.
        track_timing (bool, optional): Whether to track timing of sound playing. Defaults to False.
        sound_callback (optional): Callback for sound triggers. Defaults to None.
        sound_load_buffer_time (float, optional): Time to wait after loading file before playing. Defaults to 0.5.
        experiment_clock (optional): Clock to get time of sound stimuli. Defaults to None.
        trigger_name (Optional[str], optional): Name of the sound trigger. Defaults to None.
        timing (list, optional): List of triggers in the form of trigger name, trigger timing. Defaults to [].

    Returns:
        list: Timing information for sound triggers.
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
    """Cycle through sound files (.wav) in a directory and return the path to the next sound file on each iteration.

    Args:
        directory (str): Path to the directory containing .wav files.

    Returns:
        Iterator[str]: Iterator that infinitely cycles through the filenames.
    """
    if not path.isdir(directory):
        error_message = f'Invalid directory=[{directory}] for sound files.'
        log.error(error_message)
        raise BciPyCoreException(error_message)
    if not directory.endswith(sep):
        directory += sep
    return itertools.cycle(glob.glob(directory + '*.wav'))


def get_fixation(is_txt: bool) -> str:
    """Return the correct stimulus fixation given the type (text or image).

    Args:
        is_txt (bool): Whether the fixation is text or image.

    Returns:
        str: Fixation stimulus (text or image path).
    """
    if is_txt:
        return DEFAULT_TEXT_FIXATION
    else:
        return DEFAULT_FIXATION_PATH
