import logging
from typing import List

from bcipy.simulator.data.data_engine import QueryFilter
from bcipy.simulator.data.sampler.base_sampler import Sampler, format_samples
from bcipy.simulator.data.trial import Trial
from bcipy.simulator.util.state import SimState

log = logging.getLogger(__name__)


class TargetNontargetSampler(Sampler):
    """Sampler that that queries based on target/non-target label."""

    def sample(self, state: SimState) -> List[Trial]:
        """Sample trials for each symbol in the display alphabet, labeling as target or non-target."""
        sample_rows = []
        for symbol in state.display_alphabet:
            filters = self.query_filters(
                symbol, is_target=(symbol == state.target_symbol))
            filtered_data = self.data_engine.query(filters, samples=1)
            sample_rows.append(filtered_data[0])

        log.debug(f"Samples:\n{format_samples(sample_rows)}")
        return sample_rows

    def query_filters(self, symbol: str, is_target: bool) -> List[QueryFilter]:
        """Expression used to query for a single sample."""
        # QueryFilter('symbol', '==', symbol)
        return [QueryFilter('target', '==', int(is_target))]
