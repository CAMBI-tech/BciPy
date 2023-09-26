"""Tests for VEP layout functions"""
import unittest

from mockito import mock

from bcipy.display.components.layout import Layout
from bcipy.display.paradigm.vep.layout import (BoxConfiguration,
                                               CheckerboardSquare,
                                               checkerboard)


class TestVEPLayout(unittest.TestCase):
    """Test VEP Layout functionality."""

    def setUp(self):
        """Set up needed items for test."""
        self.window = mock({"size": (500, 500), "units": "norm"})
        self.container = self.window

    def test_2x2_checkerboard(self):
        """Test creation of a 2x2 board"""
        squares = checkerboard(squares=4,
                               colors=('red', 'green'),
                               center=(0, 0),
                               board_size=(0.2, 0.2))
        self.assertEqual(len(squares), 4)
        self.assertTrue(all(sq.size == (0.1, 0.1) for sq in squares))
        self.assertEqual(
            [sq.pos for sq in squares], [(-0.05, 0.05), (0.05, 0.05),
                                         (0.05, -0.05), (-0.05, -0.05)],
            msg="Squares should be ordered in a zig-zag arrangement")
        self.assertEqual([sq.color for sq in squares],
                         ['red', 'green', 'red', 'green'],
                         msg="colors should alternate")

    def test_3x3_checkerboard(self):
        """Test creation of a 3x3 board"""
        colors = ('blue', 'yellow')
        squares = checkerboard(squares=9,
                               colors=colors,
                               center=(0, 0),
                               board_size=(0.48, 0.48))
        self.assertEqual(len(squares), 9)
        self.assertTrue(all(sq.size == (0.16, 0.16) for sq in squares))
        self.assertEqual(squares[0].pos, (-0.16, 0.16))
        self.assertEqual(
            squares[3].pos, (0.16, 0.0),
            msg="First item on second row should be at the far right")

        for i, square in enumerate(squares):
            self.assertEqual(colors[i % 2], square.color,
                             "Colors should alternate")

    def test_checkerboard_inverse(self):
        """Test that a Checkerboard square can invert colors"""
        colors = ('red', 'green')
        square = CheckerboardSquare(pos=(0, 0), color='red', size=(0.1, 0.1))
        self.assertEqual(
            square.inverse(colors),
            CheckerboardSquare(pos=(0, 0), color='green', size=(0.1, 0.1)))

        self.assertEqual(square.inverse_color(colors), 'green')

    def test_4_box_config_defaults(self):
        """Test box configuration"""
        full_window = Layout()
        config = BoxConfiguration(layout=full_window,
                                  num_boxes=4,
                                  height_pct=0.25,
                                  spacing_pct=0.1)
        width, height = config.box_size
        window_width, window_height = full_window.size

        self.assertEqual(window_width, 2.0)
        self.assertEqual(window_height, 2.0)
        self.assertEqual(
            width,
            0.9,
            msg="Box width should be half of the remaining width after accounting for the spacing."
        )
        self.assertEqual(
            height,
            0.5,
            msg="Box height should be height_pct of the window height")

        positions = config.positions
        self.assertEqual(len(positions), 4)
        self.assertTrue((-0.55, 0.75) in positions,
                        msg="A box should be positioned at the top left")
        self.assertTrue((0.55, 0.75) in positions,
                        msg="A box should be positioned at the top right")
        self.assertTrue((-0.55, -0.75) in positions,
                        msg="A box should be positioned at the bottom left")
        self.assertTrue((0.55, -0.75) in positions,
                        msg="A box should be positioned at the bottom right")

    def test_6_box_config_defaults(self):
        """Test box configuration"""
        full_window = Layout()
        config = BoxConfiguration(layout=full_window,
                                  num_boxes=6,
                                  height_pct=0.25,
                                  spacing_pct=0.05)
        self.assertEqual(
            config.box_size, (0.6, 0.5),
            msg="width should account for 3 boxes in a row and 2 spaces")

        positions = config.positions
        self.assertEqual(len(positions), 6)
        self.assertEqual(len([box for box in positions if box[0] == 0.0]),
                         2,
                         msg="Two boxes should be positioned in the middle")

    def test_other_box_config(self):
        """Test supported number of boxes"""
        full_window = Layout()
        with self.assertRaises(AssertionError):
            BoxConfiguration(layout=full_window,
                             num_boxes=2,
                             height_pct=0.25,
                             spacing_pct=0.05)

    def test_height_pct_validation(self):
        """Test validations"""
        full_window = Layout()
        with self.assertRaises(AssertionError):
            BoxConfiguration(layout=full_window,
                             num_boxes=4,
                             height_pct=0.6,
                             spacing_pct=0.1)
