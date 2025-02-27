"""Classes and functions related to loading and querying data to be used in a simulation."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, NamedTuple, Union, get_args, get_origin

import pandas as pd

from bcipy.core.parameters import Parameters
from bcipy.exceptions import TaskConfigurationException
from bcipy.simulator.data import data_process
from bcipy.simulator.data.data_process import (ExtractedExperimentData,
                                               RawDataProcessor)
from bcipy.simulator.data.trial import Trial, convert_trials
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME

log = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


class QueryFilter(NamedTuple):
    """Provides an API used to query a data engine for data."""
    field: str
    operator: str
    value: Any

    def is_valid(self) -> bool:
        """Check if the filter is valid."""
        # pylint: disable=no-member
        field_type = Trial.__annotations__[self.field]

        # can't check isinstance of a subscriptable type, such as Optional.
        origin = get_origin(field_type)
        if origin:
            options = get_args(field_type)
            is_correct_type = any(isinstance(self.value, ftype) for ftype in options)
        else:
            is_correct_type = isinstance(self.value, field_type)

        return self.field in Trial._fields and self.operator in self.valid_operators and is_correct_type

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

        expr = ' and '.join([self.query_condition(filt) for filt in filters])
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
