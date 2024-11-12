"""Classes and functions related to loading and querying data to be used in a simulation."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, NamedTuple, Union

import numpy as np
import pandas as pd

from bcipy.exceptions import TaskConfigurationException
from bcipy.core.parameters import Parameters
from bcipy.simulator.data import data_process
from bcipy.simulator.data.data_process import (ExtractedExperimentData,
                                               RawDataProcessor)
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME

log = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


class Trial(NamedTuple):
    """Data for a given trial (a symbol within an Inquiry).

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
    inquiry_n: int
    inquiry_pos: int
    symbol: str
    target: int
    eeg: np.ndarray  # Channels by Samples ; ndarray.shape = (channel_n, sample_n)

    def __str__(self):
        fields = [
            f"source='{self.source}'", f"inquiry_n={self.inquiry_n}",
            f"inquiry_pos={self.inquiry_pos}", f"symbol='{self.symbol}'",
            f"target={self.target}", f"eeg={self.eeg.shape}"
        ]
        return f"Trial({', '.join(fields)})"

    def __repr__(self):
        return str(self)


class QueryFilter(NamedTuple):
    """Provides an API used to query a data engine for data."""
    field: str
    operator: str
    value: Any

    def is_valid(self) -> bool:
        """Check if the filter is valid."""
        # pylint: disable=no-member
        return self.field in Trial._fields and self.operator in self.valid_operators and isinstance(
            self.value, Trial.__annotations__[self.field])

    @property
    def valid_operators(self) -> List[str]:
        """List of supported query operators"""
        return ["<", "<=", ">", ">=", "==", "!="]


class DataEngine(ABC):
    """Abstract class for an object that loads data from one or more sources,
    processes the data using a provided processor, and provides an interface
    for querying the processed data."""

    def load(self):
        """Load data from sources."""

    @property
    def trials_df(self) -> pd.DataFrame:
        """Returns a dataframe of Trial data."""

    @abstractmethod
    def query(self,
              filters: List[QueryFilter],
              samples: int = 1) -> List[Trial]:
        """Query the data."""


def convert_trials(data_source: ExtractedExperimentData) -> List[Trial]:
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
            eeg_samples = [channel[sym_i]
                           for channel in inquiry_eeg]  # (channel_n, sample_n)
            trials.append(
                Trial(source=data_source.source_dir,
                      inquiry_n=i,
                      inquiry_pos=sym_i + 1,
                      symbol=symbol,
                      target=inquiry_labels[sym_i],
                      eeg=np.array(eeg_samples)))
    return trials


class RawDataEngine(DataEngine):
    """
    Object that loads in list of session data folders and transforms data into
    a queryable data structure.
    """

    def __init__(self, source_dirs: List[str], parameters: Parameters,
                 data_processor: RawDataProcessor):
        self.source_dirs: List[str] = source_dirs
        self.parameters: Parameters = parameters

        self.data_processor = data_processor
        self.data: List[Union[ExtractedExperimentData,
                              data_process.ExtractedExperimentData]] = []
        self._trials_df = pd.DataFrame()

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
            rows = []
            for i, source_dir in enumerate(self.source_dirs):
                log.debug(f"{i+1}. {Path(source_dir).name}")
                extracted_data = self.data_processor.process(
                    source_dir, self.parameters)
                self.data.append(extracted_data)
                rows.extend(convert_trials(extracted_data))

            self._trials_df = pd.DataFrame(rows)
            log.debug("Finished loading all data")
        return self

    @property
    def trials_df(self) -> pd.DataFrame:
        """Dataframe of Trial data."""
        if not self.data_loaded:
            self.load()
        return self._trials_df.copy()

    @property
    def data_loaded(self) -> bool:
        """Check if the data has been loaded"""
        return bool(self.data)

    def query(self,
              filters: List[QueryFilter],
              samples: int = 1) -> List[Trial]:
        """Query the engine for data using one or more filters.

        Parameters
        ----------
            filters - list of query filters
            samples - number of results to return.
            check_insufficient_results - if True, raises an exception when
                there are an insufficient number of samples.

        Returns a list of Trials.
        """
        assert self.data_loaded, "Data must be loaded before querying."
        assert all(filt.is_valid()
                   for filt in filters), "Filters must all be valid"
        assert samples >= 1, "Insufficient number of samples requested"

        expr = 'and '.join([self.query_condition(filt) for filt in filters])
        filtered_data = self._trials_df.query(expr)
        if filtered_data is None or len(filtered_data) < samples:
            raise TaskConfigurationException(
                message="Not enough samples found")

        rows = filtered_data.sample(samples)
        return [Trial(*row) for row in rows.itertuples(index=False, name=None)]

    def query_condition(self, query_filter: QueryFilter) -> str:
        """Returns the string representation of of the given query condition."""
        value = query_filter.value
        if (isinstance(value, str)):
            value = f"'{value}'"
        return f"{query_filter.field} {query_filter.operator} {value}"
