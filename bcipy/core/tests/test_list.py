"""Tests for list processing utilities"""
import unittest

from bcipy.core.list import (destutter, expanded, find_index, grouper,
                             pairwise, swapped)


class TestListUtilities(unittest.TestCase):
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

    def test_grouper_incomplete_fill_default(self):
        iterable = 'ABCDEFG'
        chunk_size = 2
        fillvalue = 'X'
        response = grouper(iterable, chunk_size, fillvalue=fillvalue)
        expected = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G', 'X')]
        for resp, exp in zip(response, expected):
            self.assertEqual(resp, exp)

    def test_grouper_incomplete_fill_value_error_without_fillvalue(self):
        iterable = 'ABCDEFG'
        chunk_size = 2
        incomplete = 'fill'

        with self.assertRaises(ValueError):
            grouper(iterable, chunk_size, incomplete=incomplete)

    def test_grouper_incomplete_ignore(self):
        iterable = 'ABCDEFG'
        chunk_size = 2
        incomplete = 'ignore'
        expected = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G')]
        response = grouper(iterable, chunk_size, incomplete=incomplete)
        for resp, exp in zip(response, expected):
            self.assertEqual(resp, exp)

    def test_grouper_incomplete_value_error_unsupported_incompelte_mode_defined(
            self):
        iterable = 'ABCDEFG'
        chunk_size = 2
        incomplete = 'not_defined'

        with self.assertRaises(ValueError):
            grouper(iterable, chunk_size, incomplete=incomplete)

    def test_find_index(self):
        """Test find index of item"""
        self.assertEqual(2, find_index([1, 2, 3, 4], 3))

    def test_find_index_using_key(self):
        """Find index using the key arg"""
        item1 = dict(a=10, b=10)
        item2 = dict(a=10, b=20)
        item3 = dict(a=20, b=10)
        item4 = dict(a=20, b=20)
        self.assertEqual(
            1,
            find_index([item1, item2, item3, item4],
                       match_item=20,
                       key=lambda item: item['b']))

    def test_find_index_matching_predicate(self):
        """Find the index of the first item matching a predicate"""
        values = [5, 7, 9, 12]
        self.assertEqual(1, find_index(values, match_item=lambda val: val > 6))

    def test_swapped(self):
        """Test swapped function."""
        self.assertEqual([1, 2, 3, 4, 5], swapped([1, 2, 3, 4, 5], 0, 0))
        self.assertEqual([1, 4, 3, 2, 5], swapped([1, 2, 3, 4, 5], 1, 3))
        self.assertEqual([1, 2, 3, 4, 5], swapped([1, 2, 3, 4, 5], 4, 4))
        self.assertEqual([1, 2, 3, 4, 5], swapped([5, 2, 3, 4, 1], 0, 4))

    def test_expanded(self):
        """Test expanded function."""
        self.assertEqual([1, 2, 3, 3, 3], expanded([1, 2, 3], length=5))

    def test_pairwise(self):
        """Test pairwise iterator"""
        iterable = 'ABCDEFG'
        response = pairwise(iterable)
        expected = [('A', 'B'), ('B', 'C'), ('C', 'D'),
                    ('D', 'E'), ('E', 'F'), ('F', 'G')]
        self.assertListEqual(expected, list(response))


if __name__ == '__main__':
    unittest.main()
