from abc import ABC
from typing import Callable, List

import numpy as np
import pandas as pd

from bcipy.helpers.exceptions import TaskConfigurationException
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.symbols import alphabet
from bcipy.simulator.helpers.data_engine import RawDataEngine
from bcipy.simulator.interfaces import SimSessionState


class Sampler(ABC):
    """ Interface to generate sample of data usable by a model """

    def sample(self, state: SimSessionState):
        ...

    def set_reshaper(self, reshaper: Callable):
        ...


class SimpleLetterSampler(Sampler):

    def __init__(self, data_engine: RawDataEngine, params: List[Parameters] = None):
        self.data_engine: RawDataEngine = data_engine
        self.parameter_files: List[Parameters] = params if params else self.data_engine.parameter_files
        self.model_input_reshaper: Callable = self.__default_reshaper
        self.alphabet: List[str] = params[0].get('symbol_set') if params else alphabet()
        self.data: pd.DataFrame = self.data_engine.transform().get_data()

    def sample(self, state: SimSessionState) -> np.ndarray:
        """
            - query eeg response by letter
            - reshape for signal model input

        Return:
            nd array (n_channel, n_trial, n_sample)
        """

        inquiry_letter_subset = state.display_alphabet
        target_letter = state.target_symbol
        eeg_responses = []
        for symbol in inquiry_letter_subset:
            is_target = int(symbol == target_letter)
            filtered_data = self.data.query(f'target == {is_target} and symbol == "{symbol}"')
            if not len(filtered_data):
                raise TaskConfigurationException(message="No eeg sample found with provided data and query")

            row = filtered_data.sample(1)
            eeg_responses.append(row['eeg'])

        return self.model_input_reshaper(eeg_responses)

    def set_reshaper(self, reshaper: Callable):
        self.model_input_reshaper = reshaper

    def __default_reshaper(self, eeg_response) -> np.ndarray:

        channels_eeg = [[] for i in range(len(eeg_response[0]))]
        for t_i, trial_channels_eeg in enumerate(eeg_response):
            for c_i, channel_eeg in enumerate(trial_channels_eeg):
                channels_eeg[c_i].append(channel_eeg)

        return np.array(channels_eeg[0])

# TODO ReplaySampler