import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd

from bcipy.helpers.parameters import Parameters
from bcipy.simulator.helpers.artifact import TOP_LEVEL_LOGGER_NAME
from bcipy.simulator.helpers.signal_helpers import (EegRawDataProcessor,
                                                    ExtractedExperimentData,
                                                    RawDataProcessor)

log = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


class DataEngine(ABC):

    def load(self):
        ...

    def transform(self):
        ...

    def get_data(self):
        ...

    @abstractmethod
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
    eeg: np.ndarray  # Channels by Samples

    def __str__(self):
        fields = [
            f"source='{self.source}'", f"inquiry_n={self.inquiry_n}",
            f"inquiry_pos={self.inquiry_pos}", f"symbol='{self.symbol}'",
            f"target={self.target}", f"eeg={self.eeg.shape}"
        ]
        return f"Trial({', '.join(fields)})"

    def __repr__(self):
        return str(self)


class RawDataEngine(DataEngine):
    """
    Object that loads in list of session data folders and transforms data into sample-able DataFrame
    """

    def __init__(self,
                 source_dirs: List[str],
                 parameters: Parameters,
                 data_processor: Optional[RawDataProcessor] = None):
        self.source_dirs: List[str] = source_dirs
        self.parameters: Parameters = parameters

        self.data_processor = data_processor or EegRawDataProcessor()
        self.data: List[ExtractedExperimentData] = []
        self.schema: Optional[pd.DataFrame] = None

        self.load()

    def load(self) -> DataEngine:
        """
        Processes raw data from data folders using provided parameter files.
            - Extracts and stores trial data, stimuli, and stimuli_labels by inquiries

        Returns:
            self for chaining
        """
        if not self.data:
            log.debug(
                f"Loading data from {len(self.source_dirs)} source directories:"
            )
            for i, source_dir in enumerate(self.source_dirs):
                log.debug(f"{i+1}. {Path(source_dir).name}")
                self.data.append(
                    self.data_processor.process(source_dir, self.parameters))

            log.debug("Finished loading all data")
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
        rows = []
        for data_source in self.data:
            rows.extend(self.trials(data_source))

        self.schema = pd.DataFrame(rows)
        return self

    def trials(self, data_source: ExtractedExperimentData) -> List[Trial]:
        """Convert extracted data from a single data source to a list of Trials."""
        trials = []
        symbols_by_inquiry = data_source.symbols_by_inquiry
        labels_by_inquiry = data_source.labels_by_inquiry

        for i, inquiry_eeg in enumerate(data_source.trials_by_inquiry):
            # iterate through each inquiry
            inquiry_symbols = symbols_by_inquiry[i]
            inquiry_labels = labels_by_inquiry[i]

            for sym_i, symbol in enumerate(inquiry_symbols):
                # iterate through each symbol in the inquiry
                eeg_samples = [channel[sym_i] for channel in inquiry_eeg
                               ]  # (channel_n, sample_n)
                trials.append(
                    Trial(source=data_source.source_dir,
                          inquiry_n=i,
                          inquiry_pos=sym_i + 1,
                          symbol=symbol,
                          target=inquiry_labels[sym_i],
                          eeg=np.array(eeg_samples)))
        return trials

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

    def __init__(self,
                 source_dir: str,
                 parameters: Parameters,
                 data_processor: Optional[RawDataProcessor] = None):
        data_paths = [str(d) for d in Path(source_dir).iterdir() if d.is_dir()]
        super().__init__(data_paths, parameters, data_processor)
