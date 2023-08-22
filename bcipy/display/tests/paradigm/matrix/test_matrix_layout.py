"""Tests for Matrix layout functions"""
import unittest

from mockito import mock

from bcipy.display.components.layout import Layout
from bcipy.display.paradigm.matrix.layout import symbol_positions


class TestMatrixLayout(unittest.TestCase):
    """Tests for calculating the grid layout"""

    def setUp(self):
        """Set up needed items for test."""
        self.window = mock({"size": (500, 500), "units": "norm"})
        self.container = self.window
        self.layout = Layout(parent=self.container,
                             left=-1.0,
                             top=1.0,
                             right=1.0,
                             bottom=-1.0)

    def test_regular_grid(self):
        """Test basic properties of a regular grid"""
        row_count = 4
        col_count = 5
        positions = symbol_positions(self.layout,
                                     rows=row_count,
                                     columns=col_count)
        self.assertEqual(len(positions), 20)

        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        self.assertEqual(len(set(y_coords)), row_count)
        self.assertEqual(len(set(x_coords)), col_count)

    def test_single_row(self):
        """Test position calculations for a single row"""

        positions = symbol_positions(self.layout, rows=1, columns=10)
        self.assertEqual(len(positions), 10)

        y_coord = positions[0][1]
        self.assertEqual(y_coord, self.layout.center[1],
                         "y-values should be centered")
        self.assertTrue(all(pos[1] == y_coord for pos in positions),
                        "all coordinates have the same y-value")
        self.assertEqual(len(set(pos[0] for pos in positions)), len(positions),
                         "all x-values should be different")
        self.assertTrue(
            all(self.layout.left <= pos[0] <= self.layout.right
                for pos in positions),
            "all x-values should be within the bounds of the display")

        self.assertAlmostEqual(abs(self.layout.left - positions[0][0]),
                               abs(self.layout.right - positions[-1][0]),
                               msg="should have the same left and right gap")
        # should all be equally spaced

    def test_single_column(self):
        """Test position calculations for a single column"""
        positions = symbol_positions(self.layout, rows=10, columns=1)
        self.assertEqual(len(positions), 10)

        x_coord = positions[0][0]
        self.assertEqual(x_coord, self.layout.center[0],
                         "x-values should be centered")
        self.assertTrue(all(pos[0] == x_coord for pos in positions),
                        "all coordinates have the same x-value")
        self.assertEqual(len(set(pos[1] for pos in positions)), len(positions),
                         "all y-values should be different")
        self.assertTrue(
            all(self.layout.bottom <= pos[1] <= self.layout.top
                for pos in positions),
            "all y-values should be within the bounds of the display")

        self.assertAlmostEqual(abs(self.layout.top - positions[0][1]),
                               abs(self.layout.bottom - positions[-1][1]),
                               msg="should have the same top and bottom gap")


if __name__ == '__main__':
    unittest.main()
