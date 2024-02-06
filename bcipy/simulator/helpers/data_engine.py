import logging
import os
from abc import ABC
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd

from bcipy.config import DEFAULT_PARAMETER_FILENAME
from bcipy.helpers import load
from bcipy.helpers.list import grouper
from bcipy.helpers.parameters import Parameters
from bcipy.simulator.helpers.signal_helpers import (ExtractedExperimentData,
                                                    process_raw_data_for_model)

log = logging.getLogger(__name__)


class DataEngine(ABC):

    def load(self):
        ...

    def transform(self):
        ...

    def get_data(self):
        ...

    def get_parameters(self) -> List[Parameters]:
        ...

class Trial(NamedTuple):
    """Data for a given trial (a symbol within an Inquiry).

    TODO: add series to facilitate easier lookup in session data.

    Attrs
    -----
        source - directory of the data source
        inquiry_n - starts at 0; does not reset at each series
        inquiry_pos  - starts at 1; position in which the symbol was presented
        symbol - alphabet symbol that was presented
        target - 1 or 0 indicating a boolean of whether this was a target symbol
        eeg - EEG data associated with this trial
    """
    source: str
    # series_n: int
    inquiry_n: int
    inquiry_pos: int
    symbol: str
    target: int
    eeg: np.ndarray

class RawDataEngine(DataEngine):
    """
    Object that loads in list of session data folders and transforms data into sample-able pd.df
    """

    def __init__(self, source_dirs: List[str], parameters: Parameters):
        # TODO read parameters from source dirx
        # TODO use os.walk() and take in single source_dir that has all data folders
        self.source_dirs: List[str] = source_dirs
        self.parameters: Parameters = parameters
        self.data: Optional[List[ExtractedExperimentData]] = None
        self.trials_by_inquiry: List[
            np.ndarray] = []  # shape (i_inquiry, n_channel, m_trial, x_sample).
        self.symbols_by_inquiry: List[List] = []  # shape (i_inquiry, s_alphabet_subset)
        self.labels_by_inquiry: List[List] = []  # shape (i_inquiry, s_alphabet_subset)
        self.schema: Optional[pd.DataFrame] = None

        # TODO validate parameters

        self.load()

    def load(self) -> DataEngine:
        """
        Processes raw data from data folders using provided parameter files.
            - Extracts and stores trial data, stimuli, and stimuli_labels by inquiries

        Returns:
            self for chaining
        """
        log.debug(f"Loading data from {len(self.source_dirs)} source directories")

        self.data = [
            process_raw_data_for_model(source_dir, self.parameters)
            for source_dir in self.source_dirs
        ]

        for data_source in self.data:
            _trigger_targetness, _trigger_timing, trigger_symbols = data_source.decoded_triggers
            self.trials_by_inquiry.append(
                np.split(data_source.trials, data_source.inquiries.shape[1], 1))
            self.symbols_by_inquiry.append([list(group) for group in
                                            grouper(trigger_symbols,
                                                    self.parameters.get('stim_length'),
                                                    incomplete="ignore")])

            self.labels_by_inquiry.append(data_source.labels)

        log.info("Finished loading all data")
        return self

    def transform(self) -> DataEngine:
        """
        Organizes all data into the following schema
            - Dataframe {"inquiry_n":int, "trial_n":int, "symbol":str, "target":int, "eeg":ndarray}

        The "eeg" data for trial looks like this
            - ndarray.shape = (channel_n, sample_n)
            - a trial is an eeg response for a symbol being displayed

        Returns:
            self for chaining
        """

        # TODO store how good evidence was in session

        rows = []
        for d_i, data_source in enumerate(self.data):
            for i in range(len(self.trials_by_inquiry[d_i])):
                symbols = self.symbols_by_inquiry[d_i][i]
                inquiry_labels = self.labels_by_inquiry[d_i][i]
                inquiry_eeg = self.trials_by_inquiry[d_i][i]

                symbol_rows = []
                for t_i, symbol in enumerate(symbols):
                    channel_eeg_samples_for_t = [channel[t_i] for channel in
                                                 inquiry_eeg]  # (channel_n, sample_n)
                    symbol_rows.append(Trial(source=data_source.source_dir,
                                  inquiry_n=i,
                                  inquiry_pos=t_i + 1,
                                  symbol=symbol,
                                  target=inquiry_labels[t_i],
                                  eeg=np.array(channel_eeg_samples_for_t)))

                rows.extend(symbol_rows)

        self.schema = pd.DataFrame(rows)

        return self

    def get_data(self):
        return self.schema.copy() if self.schema is not None else None

    def get_parameters(self):
        return self.parameters


class RawDataEngineWrapper(RawDataEngine):
    """
    Data engine that assumes all data is stored within single data folder
        - single data folder contains dataDir1, dataDir2, ...
        - each data dir contains its raw_data and triggers
    """

    def __init__(self, source_dir: str, parameters: Parameters):
        data_paths = self.get_data_dirs(source_dir)
        super().__init__(data_paths, parameters)

    @staticmethod
    def get_data_dirs(source_dir: str) -> [str]:
        """
        Returns all the data dirs within the source dir
            - e.g [dataDir1Path, dataDir2Path ... ]
        """

        assert source_dir

        directories: List[str] = [str(d) for d in Path(source_dir).iterdir() if d.is_dir()]

        return directories
