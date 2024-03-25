import logging
import random
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from bcipy.helpers.exceptions import TaskConfigurationException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.log_utils import (format_sample_df,
                                               format_sample_rows)
from bcipy.simulator.helpers.state_manager import SimState

log = logging.getLogger(__name__)


class Sampler(ABC):
    """ Interface to generate sample of data usable by a model """

    def sample(self, state: SimState):
        ...

    def set_reshaper(self, reshaper: Callable):
        ...


class EEGByLetterSampler(Sampler):

    def __init__(self,
                 data_engine: RawDataEngine,
                 parameters: Parameters = None):
        self.data_engine: RawDataEngine = data_engine
        self.parameters: Parameters = self.data_engine.parameters if not parameters else parameters
        self.model_input_reshaper: Callable = self.__default_reshaper
        self.alphabet: List[str] = self.parameters.get(
            'symbol_set') if self.parameters else alphabet()
        self.data: pd.DataFrame = self.data_engine.transform().get_data()

    def sample(self, state: SimState) -> np.ndarray:
        """
            - query eeg response by letter
            - reshape for signal model input

        Return:
            nd array (n_channel, n_trial, n_sample)
        """

        inquiry_letter_subset = state.display_alphabet
        target_letter = state.target_symbol
        sample_rows = []
        for symbol in inquiry_letter_subset:
            is_target = int(symbol == target_letter)
            # filtered_data = self.data.query(f'target == {is_target} and symbol == "{symbol}"')
            filtered_data: pd.DataFrame = self.data.query(
                f'target == {is_target}')
            if filtered_data is None or len(filtered_data) == 0:
                raise TaskConfigurationException(
                    message="No eeg sample found with provided data and query")

            row = filtered_data.sample(1).iloc[0]
            sample_rows.append(row)

        log.debug(f"EEG Samples: \n {format_sample_rows(sample_rows)}")

        eeg_responses = [r['eeg'] for r in sample_rows]
        sample = self.model_input_reshaper(eeg_responses)

        return sample

    def set_reshaper(self, reshaper: Callable):
        self.model_input_reshaper = reshaper

    def __default_reshaper(self,
                           eeg_responses: List[np.ndarray]) -> np.ndarray:
        # returns (channel_n, trial_n, sample_n)

        channels_eeg = [[] for i in range(len(eeg_responses[0]))]

        for t_i, trial_channels_eeg in enumerate(eeg_responses):
            for c_i, channel_eeg in enumerate(trial_channels_eeg):
                channels_eeg[c_i].append(channel_eeg)

        # make sure this returns (7, 10, 90) for tab_test_dynamic
        return np.array(channels_eeg)


def default_reshaper(eeg_responses: List[np.ndarray]) -> np.ndarray:
    """Default data reshaper.

    Returns
    -------
    ndarray with shape (channel_n, trial_n, sample_n)
    """

    channels_eeg = [[] for i in range(len(eeg_responses[0]))]

    for _i, trial_channels_eeg in enumerate(eeg_responses):
        for c_i, channel_eeg in enumerate(trial_channels_eeg):
            channels_eeg[c_i].append(channel_eeg)

    return np.array(channels_eeg)


class InquirySampler(Sampler):
    """Samples an inquiry of data at a time."""

    def __init__(self, data_engine: RawDataEngine):
        self.model_input_reshaper = default_reshaper
        self.data: pd.DataFrame = data_engine.transform().get_data()

        # map source to inquiry numbers
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

    def sample(self, state: SimState) -> np.ndarray:
        """Samples a random inquiry for a random data source.
        
        Ensures that if a target is shown in the current inquiry the sampled
        data will be from an inquiry where the target was presented.

        Returns
        -------
            nd array (n_channel, n_trial, n_sample)
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
            inquiry_target_index = inquiry_df[inquiry_df['target'] == 1].index[0]
            inquiry_target_position = inquiry_df.index.tolist().index(inquiry_target_index)

        if inquiry_target_position == target_position:
            # target already in the correct location, no need to adjust
            new_target_position = inquiry_target_position
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
        log.debug(f"EEG Samples: \n {format_sample_df(sorted_inquiry_df)}")

        rows = [
            sorted_inquiry_df.iloc[i] for i in range(len(sorted_inquiry_df))
        ]
        eeg_responses = [r['eeg'] for r in rows]
        return self.model_input_reshaper(eeg_responses)

    def set_reshaper(self, reshaper: Callable):
        self.model_input_reshaper = reshaper
