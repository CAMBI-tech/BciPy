import logging
from typing import List, Optional

from bcipy.simulator.data.data_engine import QueryFilter, RawDataEngine
from bcipy.simulator.data.sampler.target_nontarget_sampler import \
    TargetNontargetSampler

log = logging.getLogger(__name__)


class InquiryRangeSampler(TargetNontargetSampler):
    """Sampler that queries a subset of inquiries for each series.

    Parameters
    ----------
        data_engine - data repository to be queried
        inquiry_start - index of first inquiry to include (starting at 1)
        inquiry_count - number of inquiries to include
        series_start - index of first series to include (starting at 1)
        series_end - optional index of last series to include
    """

    def __init__(self,
                 data_engine: RawDataEngine,
                 inquiry_start: int = 1,
                 inquiry_count: int = 3,
                 series_start: int = 1,
                 series_count: Optional[int] = None):
        super().__init__(data_engine)
        if inquiry_start < 1:
            raise ValueError("inquiry_start can't be less than 1")
        if inquiry_count < 1:
            raise ValueError("inquiry_count must be at least 1")

        self.inquiry_start = inquiry_start
        self.inquiry_count = inquiry_count
        self.series_start = series_start
        self.series_count = series_count

    def query_filters(self, symbol: str, is_target: bool) -> List[QueryFilter]:
        """Expression used to query for a single sample."""
        # In the data, inquiries are 0-indexed.
        inq_start = self.inquiry_start - 1
        inq_end = inq_start + self.inquiry_count
        filters = [
            QueryFilter('target', '==', int(is_target)),
            QueryFilter('series_inquiry', '>=', inq_start),
            QueryFilter('series_inquiry', '<', inq_end),
            QueryFilter('series', '>=', self.series_start)
        ]
        if self.series_count:
            series_end = self.series_start + self.series_count
            filters.append(QueryFilter('series', '<', series_end))
        return filters

    def __str__(self):
        fields = [
            f"inquiry_start={self.inquiry_start}",
            f"inquiry_count={self.inquiry_count}",
            f"series_start={self.series_start}",
            f"series_count={self.series_count}"
        ]
        return f"InquiryRangeSampler({', '.join(fields)})"

    def __repr__(self):
        return str(self)
