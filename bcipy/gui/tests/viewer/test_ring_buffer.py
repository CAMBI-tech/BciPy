"""Tests for the RingBuffer data structure"""
import unittest

from bcipy.gui.viewer.ring_buffer import RingBuffer


class TestRingBuffer(unittest.TestCase):
    """Tests for the RingBuffer data structure"""

    def test_append(self):
        """Test append"""

        buf = RingBuffer(3)
        buf.append(1)
        self.assertEqual([1], buf.get())
        self.assertEqual([1], buf.data)
        buf.append(2)
        self.assertEqual([1, 2], buf.get())
        self.assertEqual([1, 2], buf.data)
        buf.append(3)
        self.assertEqual([1, 2, 3], buf.get())
        self.assertEqual([1, 2, 3], buf.data)
        buf.append(4)
        self.assertEqual([4, 2, 3], buf.data,
                         'Oldest data should be overwritten')
        self.assertEqual([2, 3, 4], buf.get(),
                         ('Data should be returned in order from oldest ',
                          'to newest'))

    def test_max(self):
        """RingBuffer should never exceed the max size."""
        size = 10
        buf = RingBuffer(size)

        for i in range(100):
            buf.append(i)

        self.assertEqual(size, len(buf.data))

    def test_pre_allocation(self):
        """Test pre-allocated data."""

        buf = RingBuffer(3, pre_allocated=True)
        buf.append(1)
        self.assertEqual([1, None, None], buf.get())
        self.assertEqual([1, None, None], buf.data)
        buf.append(2)
        self.assertEqual([1, 2, None], buf.get())
        self.assertEqual([1, 2, None], buf.data)
        buf.append(3)
        self.assertEqual([1, 2, 3], buf.get())
        self.assertEqual([1, 2, 3], buf.data)
        buf.append(4)
        self.assertEqual([4, 2, 3], buf.data,
                         'Oldest data should be overwritten')
        self.assertEqual([2, 3, 4], buf.get(),
                         ('Data should be returned in order from oldest ',
                          'to newest when full'))

    def test_empty_check(self):
        """Test empty check"""
        buf = RingBuffer(3, pre_allocated=True)
        self.assertTrue(buf.is_empty())
        buf.append(1)
        self.assertFalse(buf.is_empty())

        buf = RingBuffer(3, pre_allocated=False)
        self.assertTrue(buf.is_empty())
        buf.append(1)
        self.assertFalse(buf.is_empty())

        buf = RingBuffer(3, pre_allocated=True, empty_value=0)
        self.assertTrue(buf.is_empty())
        buf.append(0)
        self.assertTrue(buf.is_empty())

    def test_empty_value(self):
        """Test pre-allocated data with empty value."""

        buf = RingBuffer(3, pre_allocated=True, empty_value=0.0)
        buf.append(1.0)
        self.assertEqual([1.0, 0.0, 0.0], buf.get())
        self.assertEqual([1, 0.0, 0.0], buf.data)
        buf.append(2.0)
        self.assertEqual([1.0, 2.0, 0.0], buf.get())
        self.assertEqual([1.0, 2.0, 0.0], buf.data)
        buf.append(3.0)
        self.assertEqual([1.0, 2.0, 3.0], buf.get())
        self.assertEqual([1.0, 2.0, 3.0], buf.data)


if __name__ == '__main__':
    unittest.main()
