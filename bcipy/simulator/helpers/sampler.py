import logging
import random
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from bcipy.simulator.helpers.data_engine import (QueryFilter, RawDataEngine,
                                                 Trial)
from bcipy.simulator.helpers.log_utils import format_samples
from bcipy.simulator.helpers.state import SimState

log = logging.getLogger(__name__)


def default_reshaper(eeg_responses: List[np.ndarray]) -> np.ndarray:
    """Default data reshaper.

    Returns
    -------
    ndarray with shape (channel_n, trial_n, sample_n)
    """

    channels_eeg: List[List[np.ndarray]] = [
        [] for i in range(len(eeg_responses[0]))
    ]

    for _i, trial_channels_eeg in enumerate(eeg_responses):
        for c_i, channel_eeg in enumerate(trial_channels_eeg):
            channels_eeg[c_i].append(channel_eeg)

    return np.array(channels_eeg)


class Sampler(ABC):
    """Represents a strategy for sampling signal model data from a DataEngine
    comprised of signal data from one or more data collection sessions."""

    def __init__(self, data_engine: RawDataEngine):
        self.data_engine: RawDataEngine = data_engine
        self.model_input_reshaper: Callable = default_reshaper

    def sample(self, state: SimState) -> List[Trial]:
        """
        Query the data engine for a list of trials corresponding to each
        currently displayed symbol.

        Parameters
        ----------
            state - specifies the target symbol and current inquiry (displayed symbols).

        Returns
        -------
            a list of Trials with an item for each symbol in the current inquiry.
        """
        raise NotImplementedError

    def sample_data(self, state: SimState) -> np.ndarray:
        """
        Query for trials and reshape for signal model input according to the
        provided reshaper.

        Return:
            ndarray of shape (n_channel, n_trial, n_sample)
        """
        trials = self.sample(state)
        return self.reshaped(trials)

    def sample_with_context(self,
                            state: SimState) -> Tuple[np.ndarray, List[Trial]]:
        """
        Returns
        -------
            A tuple of the reshaped data (ndarray of shape (n_channel, n_trial, n_sample)) as
            well as a list of Trial data (metadata and data not reshaped) for context.
        """
        trials = self.sample(state)
        data = self.reshaped(trials)
        return data, trials

    def reshaped(self, sample_rows: List[Trial]) -> np.ndarray:
        """Returns the queried trials reshaped into a format that a model can predict."""
        return self.model_input_reshaper([trial.eeg for trial in sample_rows])

    def set_reshaper(self, reshaper: Callable):
        """Set the reshaper"""
        self.model_input_reshaper = reshaper

    def __str__(self):
        return f"<{self.__class__.__name__}>"


class TargetNontargetSampler(Sampler):
    """Sampler that that queries based on target/non-target label."""

    def sample(self, state: SimState) -> List[Trial]:
        sample_rows = []
        for symbol in state.display_alphabet:
            filters = self.query_filters(
                symbol, is_target=(symbol == state.target_symbol))
            filtered_data = self.data_engine.query(filters, samples=1)
            sample_rows.append(filtered_data[0])

        log.debug(f"Samples:\n{format_samples(sample_rows)}")
        return sample_rows

    def query_filters(self, symbol: str, is_target: bool) -> List[QueryFilter]:
        """Expression used to query for a single sample."""
        # QueryFilter('symbol', '==', symbol)
        return [QueryFilter('target', '==', int(is_target))]


class InquirySampler(Sampler):
    """Samples an inquiry of data at a time."""

    def __init__(self, data_engine: RawDataEngine):
        super().__init__(data_engine)

        # map source to inquiry numbers
        self.data = self.data_engine.trials_df
        self.target_inquiries, self.no_target_inquiries = self.prepare_data(
            self.data)

    def prepare_data(
        self, data: pd.DataFrame
    ) -> Tuple[Dict[Path, List[int]], Dict[Path, List[int]]]:
        """Partition the data into those inquiries that displayed the target
        and those that did not. The resulting data structures map the data
        source with a list of inquiry_n numbers."""

        target_inquiries = defaultdict(list)
        no_target_inquiries = defaultdict(list)

        # TODO: there is probably a more optimal way to use pandas to compute.
        # Look into groupby.
        for source in data['source'].unique():
            source_df = data[data['source'] == source]
            for n in source_df['inquiry_n'].unique():
                inquiry_df = source_df[source_df['inquiry_n'] == n]
                has_target = inquiry_df[inquiry_df['target'] ==
                                        1]['target'].any()

                if has_target:
                    target_inquiries[source].append(n)
                else:
                    no_target_inquiries[source].append(n)

        return target_inquiries, no_target_inquiries

    def sample(self, state: SimState) -> List[Trial]:
        """Samples a random inquiry for a random data source.

        Ensures that if a target is shown in the current inquiry the sampled
        data will be from an inquiry where the target was presented.

        Parameters
        ----------
            state - specifies the target symbol and current inquiry (displayed symbols).

        Returns
        -------
            a list of Trials with an item for each symbol in the current
            inquiry.
        """

        target_letter = state.target_symbol
        inquiry_letter_subset = state.display_alphabet

        if target_letter in inquiry_letter_subset:
            source_inquiries = self.target_inquiries
            target_position = inquiry_letter_subset.index(target_letter) + 1
        else:
            source_inquiries = self.no_target_inquiries
            target_position = 0  # unused

        # randomly select a valid data source and inquiry
        data_source = random.choice(list(source_inquiries.keys()))
        inquiry_n = random.choice(source_inquiries[data_source])

        # select all trials for the data_source and inquiry
        inquiry_df = self.data.loc[(self.data['source'] == data_source)
                                   & (self.data['inquiry_n'] == inquiry_n)]
        assert len(inquiry_df) == len(
            inquiry_letter_subset), f"Invalid data source {data_source}"

        # The inquiry may need to be re-ordered to ensure that the target is in
        # the correct position. We do this by sorting the dataframe on a custom
        # column that is the inquiry_pos for all non-targets, and the
        # new_target_position for the target.
        inquiry_target_position = target_position
        if target_letter in inquiry_letter_subset:
            inquiry_target_index = inquiry_df[inquiry_df['target'] ==
                                              1].index[0]
            inquiry_target_position = inquiry_df.index.tolist().index(
                inquiry_target_index)

        if inquiry_target_position == target_position:
            # target already in the correct location, no need to adjust
            new_target_position = inquiry_target_position + 0.0
        elif inquiry_target_position < target_position:
            # target in the inquiry needs to be pushed later in the inquiry;
            # There is another symbol currently at that position, so we need
            # to ensure that the target is past that symbol (by adding 0.5).
            new_target_position = target_position + 0.5
        elif inquiry_target_position > target_position:
            # the target in the inquiry needs to be moved earlier in the inquiry;
            # there is another non-target symbol currently at that position so
            # the target needs to be moved before that symbol.
            new_target_position = target_position - 0.5

        # define a sort column to move the target in the inquiry to the desired index
        sort_pos = inquiry_df['inquiry_pos'].where(inquiry_df['target'] == 0,
                                                   new_target_position)

        # select the inquiry items in sorted position
        sorted_inquiry_df = inquiry_df.loc[sort_pos.sort_values().index]
        rows = [
            Trial(**sorted_inquiry_df.iloc[i])
            for i in range(len(sorted_inquiry_df))
        ]
        log.debug(f"EEG Samples:\n{format_samples(rows)}")
        return rows
