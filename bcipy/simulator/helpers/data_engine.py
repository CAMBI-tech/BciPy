from abc import ABC
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from bcipy.config import DEFAULT_PARAMETER_FILENAME
from bcipy.helpers import load
from bcipy.helpers.list import grouper
from bcipy.helpers.parameters import Parameters
from bcipy.simulator.helpers.signal_helpers import ExtractedExperimentData, process_raw_data_for_model


class DataEngine(ABC):

    def load(self):
        ...

    def transform(self):
        ...

    def get_data(self):
        ...


class RawDataEngine(DataEngine):
    """
    Object that loads in list of previous session data folders and transforms data into implemented schema
    """

    def __init__(self, source_dirs: List[str]):
        # TODO read parameters from source dirx
        # TODO maybe rely upon single source of data, write some scripts that clean data and group them
        self.source_dirs: List[str] = source_dirs
        self.parameter_files: List[Parameters] = []
        self.data: Optional[List[ExtractedExperimentData]] = None
        self.trials_by_inquiry: List[np.ndarray] = []  # shape (i_inquiry, n_channel, m_trial, x_sample).
        self.symbols_by_inquiry: List[List] = []  # shape (i_inquiry, s_alphabet_subset)
        self.labels_by_inquiry: List[List] = []  # shape (i_inquiry, s_alphabet_subset)
        self.schema: Optional[pd.DataFrame] = None
        self.load()

    def load(self) -> DataEngine:
        """
        Processes raw data from data folders using provided parameter files.
            - Extracts and stores trial data, stimuli, and stimuli_labels by inquiries

        Returns:
            self for chaining
        """
        self.parameter_files = [load.load_json_parameters(str(Path(data_folder, DEFAULT_PARAMETER_FILENAME)), value_cast=True) for data_folder in self.source_dirs]

        assert len(self.source_dirs) == len(self.parameter_files)  # TODO more general parameter validation function

        self.data = [process_raw_data_for_model(source_dir, parameter) for source_dir, parameter in
                     zip(self.source_dirs, self.parameter_files)]

        for data_source, parameter in zip(self.data, self.parameter_files):
            trigger_targetness, trigger_timing, trigger_symbols = data_source.decoded_triggers
            self.trials_by_inquiry.append(np.split(data_source.trials, data_source.inquiries.shape[1], 1))
            self.symbols_by_inquiry.append([list(group) for group in
                                            grouper(trigger_symbols, parameter.get('stim_length'),
                                                    incomplete="ignore")])

            self.labels_by_inquiry.append(data_source.labels)

        return self

    def transform(self) -> DataEngine:
        """
        Organizes all data into the following schema
            - Dataframe {"series_n":int, "inquiry_n":int, "trial_n":int, "symbol":str, "target":int, "eeg":ndarray}

        The "eeg" data for trial looks like this
            - ndarray.shape = (channel_n, sample_n)
            - a trial is an eeg response for a symbol being displayed

        Returns:
            self for chaining
        """

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

# TODO ReplaySessionDataEngine
