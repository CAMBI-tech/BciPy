"""Tests for inquiry range sampler"""
import unittest
from unittest.mock import Mock

from bcipy.simulator.data.data_engine import QueryFilter
from bcipy.simulator.data.sampler.inquiry_range_sampler import \
    InquiryRangeSampler


class InquiryRangeSamplerTest(unittest.TestCase):
    """Tests for InquiryRangeSampler"""

    def test_init(self):
        """Test initialization"""
        sampler = InquiryRangeSampler(data_engine=Mock(), inquiry_count=3)

        self.assertEqual(1, sampler.inquiry_start)
        self.assertEqual(3, sampler.inquiry_count)

        self.assertRaises(
            ValueError, lambda: InquiryRangeSampler(
                data_engine=Mock(), inquiry_count=3, inquiry_start=-1))

        self.assertRaises(
            ValueError, lambda: InquiryRangeSampler(
                data_engine=Mock(), inquiry_count=0, inquiry_start=2))

    def test_query_filters(self):
        """Test that queries are limited to the configured range"""
        sampler = InquiryRangeSampler(data_engine=Mock(), inquiry_count=3)

        filters = sampler.query_filters(symbol='A', is_target=True)

        target_filter = QueryFilter('target', '==', 1)
        nontarget_filter = QueryFilter('target', '==', 0)
        start_filter = QueryFilter('series_inquiry', '>=', 0)
        end_filter = QueryFilter('series_inquiry', '<', 3)
        series_start_filter = QueryFilter('series', '>=', 1)

        self.assertEqual(4, len(filters))
        self.assertTrue(target_filter in filters)
        self.assertFalse(nontarget_filter in filters)
        self.assertTrue(start_filter in filters)
        self.assertTrue(end_filter in filters)
        self.assertTrue(series_start_filter in filters)

        filters = sampler.query_filters(symbol='A', is_target=False)
        self.assertFalse(target_filter in filters)
        self.assertTrue(nontarget_filter in filters)

    def test_filters_with_series_count(self):
        """Test when a series count is provided."""
        sampler = InquiryRangeSampler(data_engine=Mock(),
                                      inquiry_count=3,
                                      series_start=2,
                                      series_count=2)
        series_start_filter = QueryFilter('series', '>=', 2)
        series_end_filter = QueryFilter('series', '<', 4)
        filters = sampler.query_filters(symbol='A', is_target=True)
        self.assertEqual(5, len(filters))
        self.assertTrue(series_start_filter in filters)
        self.assertTrue(series_end_filter in filters)

    def test_filter_with_nondefault_start(self):
        """Test with non-defaults"""
        sampler = InquiryRangeSampler(data_engine=Mock(),
                                      inquiry_start=5,
                                      inquiry_count=1)
        start_filter = QueryFilter('series_inquiry', '>=', 4)
        end_filter = QueryFilter('series_inquiry', '<', 5)
        filters = sampler.query_filters(symbol='A', is_target=True)

        self.assertTrue(start_filter in filters)
        self.assertTrue(end_filter in filters)


if __name__ == '__main__':
    unittest.main()
