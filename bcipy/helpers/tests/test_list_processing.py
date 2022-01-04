"""Tests for list processing utilities"""
import unittest
from bcipy.helpers.list_processing import destutter


class TestListProcessing(unittest.TestCase):
    """Main test class for list processing"""

    def test_destutter(self):
        """Test removing sequential elements from a list."""
        item1 = dict(a=1, b=1)
        item2 = dict(a=1, b=2)
        item3 = dict(a=2, b=1)
        item4 = dict(a=2, b=2)

        self.assertEqual(destutter([item1, item1, item2, item3, item4]),
                         [item1, item2, item3, item4])
        self.assertEqual(
            destutter([item1, item1, item2, item3, item4],
                      key=lambda x: x['a']), [item2, item4])
        self.assertEqual(
            destutter([item1, item1, item2, item3, item4],
                      key=lambda x: x['b']), [item1, item2, item3, item4])
