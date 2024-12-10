"""Tests for list processing utilities"""
import unittest
from bcipy.core.list import destutter, grouper


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

    def test_grouper_incomplete_value_error_unsupported_incompelte_mode_defined(self):
        iterable = 'ABCDEFG'
        chunk_size = 2
        incomplete = 'not_defined'

        with self.assertRaises(ValueError):
            grouper(iterable, chunk_size, incomplete=incomplete)


if __name__ == '__main__':
    unittest.main()
