import logging
from abc import ABC
from typing import Callable, List

import numpy as np
import pandas as pd

from bcipy.helpers.exceptions import TaskConfigurationException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.helpers.log_utils import format_sample_rows
from bcipy.simulator.helpers.state_manager import SimState

log = logging.getLogger(__name__)


class Sampler(ABC):
    """ Interface to generate sample of data usable by a model """

    def sample(self, state: SimState):
        ...

    def set_reshaper(self, reshaper: Callable):
        ...


class EEGByLetterSampler(Sampler):

    def __init__(self, data_engine: RawDataEngine, params: List[Parameters] = None):
        self.data_engine: RawDataEngine = data_engine
        self.parameter_files: List[
            Parameters] = params if params else self.data_engine.parameter_files
        self.model_input_reshaper: Callable = self.__default_reshaper
        self.alphabet: List[str] = params[0].get('symbol_set') if params else alphabet()
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
            filtered_data: pd.DataFrame = self.data.query(f'target == {is_target}')
            if filtered_data is None or len(filtered_data) == 0:
                raise TaskConfigurationException(
                    message="No eeg sample found with provided data and query")

            row = filtered_data.sample(1)
            sample_rows.append(row)

        log.debug(f"EEG Samples: \n {format_sample_rows(sample_rows)}")
        eeg_responses = [r['eeg'].to_numpy()[0] for r in sample_rows]
        sample = self.model_input_reshaper(eeg_responses)

        return sample

    def set_reshaper(self, reshaper: Callable):
        self.model_input_reshaper = reshaper

    def __default_reshaper(self, eeg_responses: List[np.ndarray]) -> np.ndarray:
        # returns (channel_n, trial_n, sample_n)

        channels_eeg = [[] for i in range(len(eeg_responses[0]))]

        for t_i, trial_channels_eeg in enumerate(eeg_responses):
            for c_i, channel_eeg in enumerate(trial_channels_eeg):
                channels_eeg[c_i].append(channel_eeg)

        # TODO make sure this returns (7, 10, 90)
        return np.array(channels_eeg)

# TODO ReplaySampler
