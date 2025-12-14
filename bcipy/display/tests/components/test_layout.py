"""Tests for component layouts"""
import unittest

from mockito import mock

from bcipy.display.components.layout import (Alignment, Layout, at_bottom,
                                             at_top, centered, envelope,
                                             from_envelope, scaled_height,
                                             scaled_size, scaled_width)


class TestLayout(unittest.TestCase):
    """Test Layout functionality."""

    def setUp(self):
        """Set up needed items for test."""
        self.window = mock({"size": (500, 500), "units": "norm"})
        self.container = self.window

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

    def test_default_layout(self):
        """Test that default layout is fullscreen"""
        layout = Layout()
        self.assertEqual(1.0, layout.top)
        self.assertEqual(1.0, layout.right)
        self.assertEqual(-1.0, layout.bottom)
        self.assertEqual(-1.0, layout.left)
        self.assertEqual((0.0, 0.0), layout.center)

    def test_invariants(self):
        """Test invariant checks"""
        with self.assertRaises(AssertionError, msg="Left prop out of range"):
            Layout(parent=self.container,
                   left=-1.1,
                   top=1.0,
                   right=1.0,
                   bottom=-1.0)

        with self.assertRaises(AssertionError, msg="Right prop out of range"):
            Layout(parent=self.container,
                   left=-1.0,
                   top=1.0,
                   right=1.1,
                   bottom=-1.0)

        with self.assertRaises(AssertionError, msg="Top prop out of range"):
            Layout(parent=self.container,
                   left=-1.0,
                   top=1.1,
                   right=1.0,
                   bottom=-1.0)

        with self.assertRaises(AssertionError, msg="Bottom prop out of range"):
            Layout(parent=self.container,
                   left=-1.0,
                   top=1.0,
                   right=1.0,
                   bottom=-1.1)

    def test_position_at_bottom(self):
        """Test at_bottom constructor"""
        layout = at_bottom(self.container, height=0.1)
        self.assertEqual(-1.0, layout.left)
        self.assertEqual(1.0, layout.right)
        self.assertEqual(-0.9, layout.top)
        self.assertEqual(-1.0, layout.bottom)

    def test_centered(self):
        """Test constructor for centered layout"""
        layout = centered(width_pct=0.7, height_pct=0.8)
        self.assertEqual(-0.7, layout.left)
        self.assertEqual(0.7, layout.right)
        self.assertEqual(0.8, layout.top)
        self.assertEqual(-0.8, layout.bottom)

    def test_resize_width_center_aligned(self):
        """Test width resize with center alignment"""
        layout = Layout()
        layout.resize_width(0.5, alignment=Alignment.CENTERED)
        self.assertEqual(-0.5, layout.left)
        self.assertEqual(0.5, layout.right)

    def test_resize_width_left_aligned(self):
        """Test width resize with left alignment"""
        layout = Layout()
        layout.resize_width(0.5, alignment=Alignment.LEFT)
        self.assertEqual(-1.0, layout.left)
        self.assertEqual(0.0, layout.right)

    def test_resize_width_right_aligned(self):
        """Test width resize with right alignment"""
        layout = Layout()
        layout.resize_width(0.5, alignment=Alignment.RIGHT)
        self.assertEqual(0.0, layout.left)
        self.assertEqual(1.0, layout.right)

    def test_resize_width_larger(self):
        """Test increase in width"""
        layout = Layout(left=-0.5, right=0.5)
        layout.resize_width(2.0, alignment=Alignment.CENTERED)
        self.assertEqual(-1.0, layout.left)
        self.assertEqual(1.0, layout.right)

    def test_resize_height_center_aligned(self):
        """Test height resize with center alignment"""
        layout = Layout()
        layout.resize_height(0.5, alignment=Alignment.CENTERED)
        self.assertEqual(-0.5, layout.bottom)
        self.assertEqual(0.5, layout.top)

    def test_resize_height_top_aligned(self):
        """Test height resize with top alignment"""
        layout = Layout()
        layout.resize_height(0.5, alignment=Alignment.TOP)
        self.assertEqual(1.0, layout.top)
        self.assertEqual(0.0, layout.bottom)

    def test_resize_height_bottom_aligned(self):
        """Test height resize with bottom alignment"""
        layout = Layout()
        layout.resize_height(0.5, alignment=Alignment.BOTTOM)
        self.assertEqual(0.0, layout.top)
        self.assertEqual(-1.0, layout.bottom)

    def test_envelope(self):
        """Test calculation of a shape's envelope"""
        pos = (0, 0)
        width = 0.2
        height = 0.1
        verts = envelope(pos=pos, size=(width, height))
        self.assertEqual(len(verts), 4)
        self.assertTrue((-0.1, 0.05) in verts)
        self.assertTrue((-0.1, -0.05) in verts)
        self.assertTrue((0.1, -0.05) in verts)
        self.assertTrue((0.1, 0.05) in verts)

    def test_from_envelope(self):
        """Test constructing a layout from an envelope"""
        verts = envelope(pos=(0, 0), size=(0.2, 0.1))
        layout = from_envelope(verts)
        self.assertEqual(layout.top, 0.05)
        self.assertEqual(layout.bottom, -0.05)
        self.assertEqual(layout.left, -0.1)
        self.assertEqual(layout.right, 0.1)
        self.assertEqual(layout.center, (0, 0))

    def test_scaled_size(self):
        """Test scaling the size to make shapes that display as squares"""
        self.assertEqual(
            scaled_size(height=0.2, window_size=(500, 500)), (0.2, 0.2),
            msg="Height and width should be the same in a square window")
        self.assertEqual(
            scaled_size(height=0.2, window_size=(800, 500)), (0.125, 0.2),
            msg="Width should be proportional to the window aspect")

        self.assertEqual(
            scaled_size(height=0.2, window_size=(
                800, 500), units='height'), (0.2, 0.2),
            msg="Width should be the same in 'height' units")

    def test_scaled_height(self):
        """Test calculation of scaled height based on width"""
        self.assertEqual(scaled_height(0.125, window_size=(800, 500)), 0.2)

    def test_scaled_width(self):
        """Test calculation of a scaled width based on height"""
        self.assertEqual(scaled_width(height=0.2, window_size=(500, 500)), 0.2)
        self.assertEqual(scaled_width(height=0.2, window_size=(800, 500)),
                         0.125)


if __name__ == '__main__':
    unittest.main()
