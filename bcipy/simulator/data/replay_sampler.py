import logging
from pathlib import Path
from typing import Dict, List, Optional

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.helpers.session import read_session
from bcipy.simulator.data.data_engine import QueryFilter, RawDataEngine, Trial
from bcipy.simulator.data.sampler import Sampler, format_samples
from bcipy.simulator.util.state import SimState
from bcipy.task.data import Session

log = logging.getLogger(__name__)


def compute_inquiry_index(series_counts: List[int], series: int,
                          inquiry: int) -> int:
    """Compute the inquiry index out of all inquiries in the session.

    Parameters
    ----------
        series_counts - number of inquiries per series for a given session
        series - the series of interest (1-indexed)
        inquiry - the inquiry relative to the start of that series (0-indexed)
    """
    assert inquiry < series_counts[series - 1], "Inquiry out of range"
    if series > 1:
        return sum(series_counts[0:series - 1]) + inquiry
    return inquiry


def compute_series_counts(session: Session) -> List[int]:
    """Returns a list of the number of inquiries for each series in the session."""
    return [len(series) for series in session.series]


def matches_task_text(session: Session, parameters: dict) -> bool:
    """Determines whether the session matches the provided parameters."""
    last_inq = session.last_inquiry()
    if last_inq:
        return last_inq.target_text == parameters['task_text']
    return False


class ReplaySampler(Sampler):
    """Sampler that provides inquiries for replaying a session.

    If a decision is made sooner than in the original session, inquiries may be
    skipped. If it takes longer to make a decision than in the original session,
    additional inquiries will be fetched from the data randomly.

    Does not handle misspellings that were not in the original session (extra series).

    Note also that the sampler requires symbols to be presented as they were in
    the original data collection.
    """

    def __init__(self, data_engine: RawDataEngine):
        super().__init__(data_engine)

        self.session_data = self.load_session_data()
        if len(self.session_data.keys()):
            log.warning("Only 1 dataset is supported.")
        self.current_source_index = 0
        self.samples_provided = 0

    @property
    def loaded_source_dirs(self) -> List[str]:
        """List of loaded source directories"""
        return [*self.session_data.keys()]

    @property
    def current_source(self) -> Optional[str]:
        """Current data source"""
        if len(self.loaded_source_dirs) > self.current_source_index:
            return self.loaded_source_dirs[self.current_source_index]
        return None

    @property
    def current_series_counts(self) -> List[int]:
        """Number of inquires in each series for the current data source."""
        if not self.current_source:
            return []
        return self.session_data.get(self.current_source, [])

    def load_session_data(self) -> Dict[str, List[int]]:
        """Load series counts for each source with a matching session file.

        Returns
        -------
            dict mapping source_dir to series counts (inquiries per series for
              that session.)
        """
        data = {}
        for source_dir in self.data_engine.source_dirs:
            session_path = Path(source_dir, SESSION_DATA_FILENAME)
            if session_path.exists():
                session = read_session(str(session_path))
                if self.should_load_session(session):
                    data[source_dir] = compute_series_counts(session)
        return data

    def should_load_session(self, session: Session) -> bool:
        """Test to check whether data for a session should be loaded."""
        return matches_task_text(session, self.data_engine.parameters)

    def sample(self, state: SimState) -> List[Trial]:
        """Sample data sequentially."""

        sample_rows = []
        for symbol in state.display_alphabet:
            filters = self.query_filters(state, symbol)
            filtered_data = self.data_engine.query(filters, samples=1)
            sample_rows.append(filtered_data[0])

        log.debug(f"Samples:\n{format_samples(sample_rows)}")
        self.update(state)
        return sample_rows

    def should_use_random_data(self, state: SimState) -> bool:
        """Check if this class should output random data."""
        if not self.current_source:
            log.info("No current data source; using random inquiry")
            return True
        if state.series_n > len(self.current_series_counts):
            log.info(
                "More series presented than in the original data; using random inquiry."
            )
            return True

        inq_count = self.current_series_counts[state.series_n - 1]
        if state.inquiry_n >= inq_count:
            log.info(
                "More inquiries required for the current series than the original; using random inquiry."
            )
            return True
        return False

    def update(self, state: SimState):
        """Update the current state"""
        self.samples_provided += 1

    def current_inquiry_num(self, state: SimState) -> Optional[int]:
        """Compute the next inquiry based on the sim state."""
        assert self.current_source, "Current data source is required."
        series_counts = self.session_data[self.current_source]
        return compute_inquiry_index(series_counts, state.series_n,
                                     state.inquiry_n)

    def query_filters(self, state: SimState, symbol: str) -> List[QueryFilter]:
        """Expression used to query for a single sample."""

        if self.should_use_random_data(state):
            is_target = (symbol == state.target_symbol)
            return [QueryFilter('target', '==', int(is_target))]

        inquiry = self.current_inquiry_num(state)
        return [
            QueryFilter('source', '==', self.current_source),
            QueryFilter('inquiry_n', '==', inquiry),
            QueryFilter('symbol', '==', symbol)
        ]
