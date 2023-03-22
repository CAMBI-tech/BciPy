"""Tests for component layouts"""
import unittest

from mockito import mock

from bcipy.display.components.layout import Layout, WindowContainer, at_top


class TestLayout(unittest.TestCase):
    """Test Layout functionality."""

    def setUp(self):
        """Set up needed items for test."""
        self.window = mock({"size": (500, 500), "units": "norm"})
        self.container = WindowContainer(win=self.window)

    def test_container(self):
        """Test container protocol"""
        self.assertEqual(self.container.size, (500, 500))
        self.assertEqual(self.container.units, "norm")

    def test_position_at_top(self):
        """Test helper function"""
        layout = at_top(self.container, height=0.1)
        self.assertEqual(-1.0, layout.left)
        self.assertEqual(1.0, layout.right)
        self.assertEqual(1.0, layout.top)
        self.assertEqual(0.9, layout.bottom)

    def test_computed_properties(self):
        """Test computed"""
        layout = Layout(parent=self.container,
                        left=-1.0,
                        top=1.0,
                        right=1.0,
                        bottom=0.9)
        self.assertEqual("norm", layout.units)
        self.assertEqual(2.0, layout.width)
        self.assertAlmostEqual(0.1, layout.height)
        self.assertEqual((0, 0.95), layout.center)
        self.assertEqual((-1.0, 0.95), layout.left_middle)
        self.assertEqual((1.0, 0.95), layout.right_middle)

    def test_full_size(self):
        """Test layout that takes the entire window."""
        layout = Layout(parent=self.container,
                        left=-1.0,
                        top=1.0,
                        right=1.0,
                        bottom=-1.0)
        self.assertEqual((0.0, 0.0), layout.center)


if __name__ == '__main__':
    unittest.main()
