import logging
from typing import List

from bcipy.simulator.data.data_engine import QueryFilter, RawDataEngine, Trial
from bcipy.simulator.data.sampler.base_sampler import Sampler, format_samples
from bcipy.simulator.exceptions import IncompatibleData
from bcipy.simulator.util.state import SimState

log = logging.getLogger(__name__)


class ReplaySampler(Sampler):
    """Sampler that provides inquiries for replaying a session.

    Assumes that the state is sending sample requests sequentially and that the
    order of symbols match the original session.
    """

    def __init__(self, data_engine: RawDataEngine):
        super().__init__(data_engine)

        source_count = len(self.data_engine.source_dirs)
        if source_count > 1:
            log.warning("Only 1 dataset is supported.")
        if source_count == 0:
            raise IncompatibleData("At least one data source is required.")
        self.current_source_index = 0

    @property
    def current_source(self) -> str:
        """Current data source"""
        return self.data_engine.source_dirs[self.current_source_index]

    def sample(self, state: SimState) -> List[Trial]:
        """Sample data that exactly matches the provided state."""

        sample_rows = []
        for symbol in state.display_alphabet:
            filters = self.query_filters(state, symbol)
            filtered_data = self.data_engine.query(filters, samples=1)
            sample_rows.append(filtered_data[0])
        log.info(f"Samples:\n{format_samples(sample_rows)}")

        return sample_rows

    def query_filters(self, state: SimState, symbol: str) -> List[QueryFilter]:
        """Expression used to query for a single sample."""
        return [
            QueryFilter('source', '==', self.current_source),
            QueryFilter('series', '==', state.series_n),
            QueryFilter('series_inquiry', '==', state.inquiry_n),
            QueryFilter('symbol', '==', symbol)
        ]

    def next_source(self) -> None:
        """Increment to query from the next data source."""
        last_index = len(self.data_engine.source_dirs) - 1
        if self.current_source_index < last_index:
            self.current_source_index += 1
        else:
            self.current_source_index = 0
