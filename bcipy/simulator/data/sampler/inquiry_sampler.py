import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bcipy.simulator.data.data_engine import RawDataEngine
from bcipy.simulator.data.sampler.base_sampler import Sampler, format_samples
from bcipy.simulator.data.trial import Trial
from bcipy.simulator.util.state import SimState

log = logging.getLogger(__name__)


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
        data will be from an inquiry where the target was presented. Note that
        the inquiry may need to be re-ordered to ensure that the target is in
        the correct position.

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

        new_target_position = float(target_position)
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
        log.info(f"EEG Samples:\n{format_samples(rows)}")
        return rows
