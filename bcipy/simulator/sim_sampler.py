import random
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bcipy.helpers.parameters import Parameters
from bcipy.simulator.interfaces import DataEngine, ModelHandler, Sampler, MetricReferee, StateManager, SessionState
from bcipy.simulator.simulator_base import Simulator
from typing import Optional, Callable, Dict, Any

import numpy as np

from bcipy.config import SESSION_DATA_FILENAME, DEFAULT_PARAMETER_FILENAME
from bcipy.helpers import load
from bcipy.helpers.list import grouper
from bcipy.helpers.session import evidence_records, read_session
from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import PcaRdaKdeModel
from bcipy.simulator.helpers.signal_helpers import process_raw_data_for_model, ExtractedExperimentData
from bcipy.simulator.helpers.sim_viz import plot_replay_comparison
from bcipy.simulator.simulator_base import Simulator, SimulatorData


class SimulatorCopyPhraseSampler(Simulator):
    """Simulator.

    Base class for BciPy Simulations.

    Requirements:
    - run closed loop simulation of {TaskType} with {data} with {simulation_params}
    """

    def __init__(self, parameter_path, save_dir, data_engine: DataEngine, model_handler: ModelHandler, sampler: Sampler,
                 state_manager: StateManager, referee: MetricReferee, source_data_path=None,
                 verbose=False):
        super(SimulatorCopyPhraseSampler, self).__init__()

        self.parameters = self.load_parameters(parameter_path)
        self.save_dir = save_dir
        self.model_handler: ModelHandler = model_handler
        self.sampler: Sampler = sampler
        self.referee: MetricReferee = referee
        self.state_manager: StateManager = state_manager

        self.data_engine: DataEngine = data_engine
        self.source_data_path = source_data_path

        self.symbol_set = alphabet()
        self.write_output = False
        self.data_loader = None

        # self.signal_models_classes = [PcaRdaKdeModel for m in self.signal_models]  # Hardcoded rn

    def run(self):
        while not self.state_manager.is_done():
            self.state_manager.add_state('display_alphabet', self.__get_inquiry_alp_subset(self.state_manager.get_state()))
            sampled_data = self.sampler.sample(self.state_manager.get_state())
            evidence = self.model_handler.generate_evidence(self.state_manager.get_state(), sampled_data)

            self.state_manager.update(evidence)

    def __get_inquiry_alp_subset(self, state: SessionState):
        # TODO put this in own file or object
        subset_length = 10
        return random.sample(self.symbol_set, subset_length)

    def load_parameters(self, path):
        # TODO validate parameters
        parameters = load.load_json_parameters(path, value_cast=True)
        sim_parameters = load.load_json_parameters(
            "bcipy/simulator/sim_parameters.json", value_cast=True)

        parameters.add_missing_items(sim_parameters)
        return parameters


class RawDataEngine(DataEngine):

    def __init__(self, source_dirs: list[str], parameters: list[Parameters]):
        self.source_dirs: list[str] = source_dirs
        self.parameters: list[Parameters] = parameters
        self.data: Optional[list[ExtractedExperimentData]] = None
        self.trials_by_inquiry: list[np.ndarray] = []  # shape (i_inquiry, n_channel, m_trial, x_sample).
        self.symbols_by_inquiry: list[list] = []  # shape (i_inquiry, s_alphabet_subset)
        self.labels_by_inquiry: list[list] = []  # shape (i_inquiry, s_alphabet_subset)
        self.schema: Optional[pd.DataFrame] = None
        self.load()

    def load(self) -> DataEngine:
        # formatting data into inquiries

        assert len(self.source_dirs) == len(self.parameters)

        self.data = [process_raw_data_for_model(source_dir, parameter) for source_dir, parameter in
                     zip(self.source_dirs, self.parameters)]

        for data_source, parameter in zip(self.data, self.parameters):
            trigger_targetness, trigger_timing, trigger_symbols = data_source.decoded_triggers
            self.trials_by_inquiry.append(np.split(data_source.trials, data_source.inquiries.shape[1], 1))
            self.symbols_by_inquiry.append([list(group) for group in
                                            grouper(trigger_symbols, parameter.get('stim_length'),
                                                    incomplete="ignore")])

            self.labels_by_inquiry.append(data_source.labels)

        return self

    def transform(self) -> DataEngine:
        # setting the schema

        cols = ["series_n", "inquiry_n", "trial_n", "symbol", "target", "eeg"]
        types = [int, int, int, str, int, np.ndarray]

        rows = []
        for d_i in range(len(self.data)):
            for i in range(len(self.trials_by_inquiry[d_i])):
                symbols = self.symbols_by_inquiry[d_i][i]
                inquiry_labels = self.labels_by_inquiry[d_i][i]
                inquiry_eeg = self.trials_by_inquiry[d_i][i]

                symbol_rows = []
                for t_i in range(len(symbols)):
                    channel_eeg_samples_for_t = [channel[t_i] for channel in inquiry_eeg]  # (channel_n, sample_n)
                    row = {'inquiry_n': i, 'trial_n': t_i, 'symbol': symbols[t_i], 'target': inquiry_labels[t_i],
                           'eeg': np.array(channel_eeg_samples_for_t)}

                    symbol_rows.append(row)

                rows.extend(symbol_rows)

        self.schema = pd.DataFrame(rows)

        return self

    def get_data(self):
        return self.schema.copy() if self.schema is not None else self.data


class SimpleLetterSampler(Sampler):

    def __init__(self, data_engine: RawDataEngine, params=None, model_reshaper: Callable = None):
        self.data_engine: RawDataEngine = data_engine
        self.params: Parameters = params if params else self.data_engine.parameters
        self.model_input_reshaper: Callable = model_reshaper if model_reshaper else self.__default_reshaper

        self.alphabet: list[str] = params.get('symbol_set') if params else alphabet()
        self.data: pd.DataFrame = self.data_engine.transform().get_data()

    def sample(self, state: SessionState) -> np.ndarray:
        """
            - randomly generate subset of letters for inquiry, shud eventually be based on language model (stored in state)
            - query eeg response by letter
            - reshape for model input

        Return:
            (n_channel, n_trial, n_sample)
        """
        # TODO make map of inquiry_n -> target_symbol, maybe

        inquiry_letter_subset = state.display_alphabet
        target_letter = state.target_symbol
        eeg_responses = []
        for symbol in inquiry_letter_subset:
            is_target = int(symbol == target_letter)
            filtered_data = self.data.query(f'target == {is_target} and symbol == "{symbol}"')
            row = filtered_data.sample(
                1) if len(filtered_data) else None  # TODO figure out what to do when no sample is found. e.g no eeg data for nontarget symbol of 'x'

            if row is None:
                print("row is None")
                breakpoint()

            eeg_responses.append(row['eeg'])

        print(inquiry_letter_subset)
        return self.model_input_reshaper(eeg_responses)

    def __default_reshaper(self, eeg_response) -> np.ndarray:

        channels_eeg = [[] for i in range(len(eeg_response[0]))]
        for t_i, trial_channels_eeg in enumerate(eeg_response):
            for c_i, channel_eeg in enumerate(trial_channels_eeg):
                channels_eeg[c_i].append(channel_eeg)

        return np.array(channels_eeg[0])

# TODO add typing hints
