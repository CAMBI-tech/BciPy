"""Defines common functionality for GUI layouts."""
from enum import Enum
from typing import List, Optional, Protocol, Tuple


class Container(Protocol):
    """Protocol for an enclosing container with units and size."""
    size: Tuple[float, float]
    units: str


# for norm units
DEFAULT_LEFT = -1.0
DEFAULT_TOP = 1.0
DEFAULT_RIGHT = 1.0
DEFAULT_BOTTOM = -1.0


class Alignment(Enum):
    """Specifies how elements should be aligned spatially"""
    CENTERED = 1
    LEFT = 2
    RIGHT = 3
    TOP = 4
    BOTTOM = 5

    @classmethod
    def horizontal(cls):
        """Subset used for horizontal alignment"""
        return [Alignment.CENTERED, Alignment.LEFT, Alignment.RIGHT]

    @classmethod
    def vertical(cls):
        """Subset used for vertical alignment"""
        return [Alignment.CENTERED, Alignment.TOP, Alignment.BOTTOM]


# Positioning functions
def above(y_coordinate: float, amount: float) -> float:
    """Returns a new y_coordinate value that is above the provided value
    by the given amount."""
    assert amount >= 0, 'Amount must be positive'
    return y_coordinate + amount


def below(y_coordinate: float, amount: float) -> float:
    """Returns a new y_coordinate value that is below the provided value
    by the given amount."""
    assert amount >= 0, 'Amount must be positive'
    return y_coordinate - amount


def right_of(x_coordinate: float, amount: float) -> float:
    """Returns a new x_coordinate value that is to the right of the
    provided value by the given amount."""
    assert amount >= 0, 'Amount must be positive'
    return x_coordinate + amount


def left_of(x_coordinate: float, amount: float) -> float:
    """Returns a new x_coordinate value that is to the left of the
    provided value by the given amount."""
    assert amount >= 0, 'Amount must be positive'
    return x_coordinate - amount


def envelope(pos: Tuple[float, float],
             size: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Compute the vertices for the envelope of a shape centered at pos with
    the given size."""
    width, height = size
    half_w = width / 2
    half_h = height / 2
    return [(left_of(pos[0], half_w), above(pos[1], half_h)),
            (right_of(pos[0], half_w), above(pos[1], half_h)),
            (right_of(pos[0], half_w), below(pos[1], half_h)),
            (left_of(pos[0], half_w), below(pos[1], half_h))]


def scaled_size(height: float,
                window_size: Tuple[float, float],
                units: str = 'norm') -> Tuple[float, float]:
    """Scales the provided height value to reflect the aspect ratio of a
    visual.Window. Used for creating squared stimulus. Returns (w,h) tuple"""
    if units == 'height':
        width = height
        return (width, height)

    win_width, win_height = window_size
    width = (win_height / win_width) * height
    return (width, height)


def scaled_height(width: float,
                  window_size: Tuple[float, float],
                  units: str = 'norm') -> float:
    """Given a width, find the equivalent height scaled to the aspect ratio of
    a window with the given size"""
    if units == 'height':
        return width
    win_width, win_height = window_size
    return width / (win_height / win_width)


def scaled_width(height: float,
                 window_size: Tuple[float, float],
                 units: str = 'norm'):
    """Given a height, find the equivalent width scaled to the aspect ratio of
    a window with the given size"""
    width, _height = scaled_size(height, window_size, units)
    return width


class Layout(Container):
    """Class with methods for positioning elements within a parent container.
    """

    def __init__(self,
                 parent: Optional[Container] = None,
                 left: float = DEFAULT_LEFT,
                 top: float = DEFAULT_TOP,
                 right: float = DEFAULT_RIGHT,
                 bottom: float = DEFAULT_BOTTOM,
                 units: float = "norm"):
        self.units = units
        self.parent = parent
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.check_invariants()

    def check_invariants(self):
        """Check that all invariants hold true."""
        # https://psychopy.org/general/units.html#units
        assert self.units in ['height',
                              'norm'], "Units must be 'height' or 'norm'"
        if self.units == "norm":
            assert (0.0 <= self.height <= 2.0), "Height must be in norm units."
            assert (0.0 <= self.width <= 2.0), "Width must be in norm units."
            assert (-1.0 <= self.top <=
                    1.0), "Top must be a y-value in norm units"
            assert (-1.0 <= self.left <=
                    1.0), "Left must be an x-value in norm units"
            assert (-1.0 <= self.bottom <=
                    1.0), "Bottom must be a y-value in norm units"
            assert (-1.0 <= self.right <=
                    1.0), "Right must be an x-value in norm units"
        if self.units == "height":
            assert (0.0 <= self.height <=
                    1.0), "Height must be in height units."
            assert (-0.5 <= self.top <=
                    0.5), "Top must be a y-value in height units"
            assert (-0.5 <= self.bottom <=
                    0.5), "Bottom must be a y-value in height units"

        if self.parent:
            assert 0 < self.width <= self.parent.size[
                0], "Width must be greater than 0 and fit within the parent width."
            assert 0 < self.height <= self.parent.size[
                1], "Height must be greater than 0 and fit within the parent height."

    def scaled_size(self, height: float) -> Tuple[float, float]:
        """Returns the (w,h) value scaled to reflect the aspect ratio of a
        visual.Window. Used for creating squared stimulus"""
        if self.units == 'height':
            width = height
            return (width, height)
        assert self.parent is not None, 'Parent must be configured'
        return scaled_size(height, self.parent.size, self.units)

    @property
    def size(self) -> Tuple[float, float]:
        """Layout size."""
        return (self.width, self.height)

    @property
    def width(self) -> float:
        """Width in norm units of this component."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Height in norm units of this component."""
        return self.top - self.bottom

    @property
    def left_top(self) -> float:
        """Top left position"""
        return (self.left, self.top)

    @property
    def right_bottom(self) -> float:
        """Bottom right position"""
        return (self.right, self.bottom)

    @property
    def horizontal_middle(self) -> float:
        """x-axis value in norm units for the midpoint of this component"""
        return (self.left + self.right) / 2

    @property
    def vertical_middle(self) -> float:
        """x-axis value in norm units for the midpoint of this component."""
        return (self.top + self.bottom) / 2

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the component in norm units. Returns a (x,y) tuple."""
        return (self.horizontal_middle, self.vertical_middle)

    @property
    def left_middle(self) -> Tuple[float, float]:
        """Point centered on the left-most edge."""
        return (self.left, self.vertical_middle)

    @property
    def right_middle(self) -> Tuple[float, float]:
        """Point centered on the right-most edge."""
        return (self.right, self.vertical_middle)

    def resize_width(self,
                     width_pct: float,
                     alignment: Alignment = Alignment.CENTERED) -> float:
        """Adjust the width of the current layout.

        Parameters
        ----------
            width_pct - percentage of the current width
            alignment - specifies how the remaining width should be aligned.
        """
        assert 0 < width_pct, 'width_pct must be greater than 0'
        assert alignment in Alignment.horizontal()
        new_width = self.width * width_pct

        left = self.left
        right = self.right

        margin = self.width - new_width
        if alignment == Alignment.CENTERED:
            half_margin = margin / 2
            left = left + half_margin
            right = right - half_margin
        elif alignment == Alignment.LEFT:
            right = right - margin
        elif alignment == Alignment.RIGHT:
            left = left + margin

        self.left = left
        self.right = right
        self.check_invariants()

    def resize_height(self,
                      height_pct: float,
                      alignment: Alignment = Alignment.CENTERED) -> float:
        """Adjust the height of the current layout.

        Parameters
        ----------
            height_pct - percentage of the current width
            alignment - specifies how the remaining width should be aligned.
        """
        assert 0 < height_pct, 'height_pct must be greater than 0'
        assert alignment in Alignment.vertical()

        new_height = self.height * height_pct

        top = self.top
        bottom = self.bottom

        margin = self.height - new_height
        if alignment == Alignment.CENTERED:
            half_margin = margin / 2
            top = top - half_margin
            bottom = bottom + half_margin
        elif alignment == Alignment.TOP:
            bottom = bottom + margin
        elif alignment == Alignment.BOTTOM:
            top = top - margin

        self.top = top
        self.bottom = bottom
        self.check_invariants()


# Factory functions
def at_top(parent: Container, height: float) -> Layout:
    """Constructs a layout of a given height that spans the full width of the
    window and is positioned at the top.

    Parameters
    ----------
        height - value in 'norm' units
    """
    top = DEFAULT_TOP
    return Layout(parent=parent,
                  left=DEFAULT_LEFT,
                  top=top,
                  right=DEFAULT_RIGHT,
                  bottom=top - height)


def at_bottom(parent: Container, height: float) -> Layout:
    """Constructs a layout of a given height that spans the full width of the
    window and is positioned at the bottom"""
    bottom = DEFAULT_BOTTOM
    return Layout(parent=parent,
                  left=DEFAULT_LEFT,
                  top=DEFAULT_BOTTOM + height,
                  right=DEFAULT_RIGHT,
                  bottom=bottom)


def centered(parent: Optional[Container] = None,
             width_pct: float = 1.0,
             height_pct: float = 1.0) -> Layout:
    """Constructs a layout that is centered on the screen. Default size is
    fullscreen but optional parameters can be used to adjust the width and
    height.

    Parameters
    ----------
        parent - optional parent
        width_pct - optional; sets the width to a given percentage of
          fullscreen.
        height_pct - optional; sets the height to a given percentage of
          fullscreen.
    """
    container = Layout(parent=parent)
    container.resize_width(width_pct, alignment=Alignment.CENTERED)
    container.resize_height(height_pct, alignment=Alignment.CENTERED)
    return container


def from_envelope(verts: List[Tuple[float, float]]) -> Layout:
    """Constructs a layout from a list of vertices which comprise a shape's
    envelope."""
    x_coords, y_coords = zip(*verts)
    return Layout(left=min(x_coords),
                  top=max(y_coords),
                  right=max(x_coords),
                  bottom=min(y_coords))


def height_units(window_size: Tuple[float, float]) -> Layout:
    """Constructs a layout with height units using the given Window
    dimensions

    for an aspect ratio of 4:3
    4 widths / 3 height = 1.333
    1.333 / 2 = 0.667
    so, left is -0.667 and right is 0.667
    """
    win_width, win_height = window_size
    right = (win_width / win_height) / 2
    return Layout(left=-right,
                  top=0.5,
                  right=right,
                  bottom=-0.5,
                  units='height')
