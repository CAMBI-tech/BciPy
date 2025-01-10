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
        sampler = InquiryRangeSampler(data_engine=Mock(), inquiry_end=3)

        self.assertEqual(0, sampler.inquiry_start)
        self.assertEqual(3, sampler.inquiry_end)

        self.assertRaises(
            AssertionError, lambda: InquiryRangeSampler(
                data_engine=Mock(), inquiry_end=3, inquiry_start=-1))

        self.assertRaises(
            AssertionError, lambda: InquiryRangeSampler(
                data_engine=Mock(), inquiry_end=1, inquiry_start=2))

    def test_query_filters(self):
        """Test that queries are limited to the configured range"""
        sampler = InquiryRangeSampler(data_engine=Mock(), inquiry_end=3)

        filters = sampler.query_filters(symbol='A', is_target=True)

        start_filter = QueryFilter('series_inquiry', '>=', 0)
        end_filter = QueryFilter('series_inquiry', '<=', 3)

        self.assertTrue(start_filter in filters)
        self.assertTrue(end_filter in filters)


if __name__ == '__main__':
    unittest.main()
