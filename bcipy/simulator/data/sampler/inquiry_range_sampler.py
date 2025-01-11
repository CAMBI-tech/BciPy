import logging
from typing import List, Optional

from bcipy.simulator.data.data_engine import QueryFilter, RawDataEngine
from bcipy.simulator.data.sampler.target_nontarget_sampler import \
    TargetNontargetSampler

log = logging.getLogger(__name__)


class InquiryRangeSampler(TargetNontargetSampler):
    """Sampler that queries a subset of inquiries for each series."""

    def __init__(self,
                 data_engine: RawDataEngine,
                 inquiry_end: int = 3,
                 inquiry_start: int = 0,
                 series_start: int = 1,
                 series_end: Optional[int] = None):
        super().__init__(data_engine)
        assert inquiry_start >= 0, "Inquiry_start can't be negative"
        assert inquiry_end >= inquiry_start, "Range definition error"
        self.inquiry_start = inquiry_start
        self.inquiry_end = inquiry_end
        self.series_start = series_start
        self.series_end = series_end

    def query_filters(self, symbol: str, is_target: bool) -> List[QueryFilter]:
        """Expression used to query for a single sample."""
        filters = [
            QueryFilter('target', '==', int(is_target)),
            QueryFilter('series_inquiry', '>=', self.inquiry_start),
            QueryFilter('series_inquiry', '<=', self.inquiry_end),
            QueryFilter('series', '>=', self.series_start)
        ]
        if self.series_end:
            filters.append(QueryFilter('series', '<=', self.series_end))
        return filters

    def __str__(self):
        fields = [
            f"inquiry_start='{self.inquiry_start}'",
            f"inquiry_end={self.inquiry_end}",
            f"series_start={self.series_start}",
            f"series_end={self.series_end}"
        ]
        return f"InquiryRangeSampler({', '.join(fields)})"

    def __repr__(self):
        return str(self)
