"""Defines common functionality for GUI layouts."""
from enum import Enum
from typing import Protocol, Tuple

from psychopy import visual


class Container(Protocol):
    """Protocol for an enclosing container with units and size."""
    win: visual.Window
    size: Tuple[float, float]
    units: str


class WindowContainer(Container):
    """Wrapper for a Window. This is not needed in Python 3.8+ if Container is
    a subclass of typing.Protocol."""

    def __init__(self, win: visual.Window):
        self.win = win

    @property
    def size(self):
        """Window size"""
        return self.win.size

    @property
    def units(self):
        """Window units"""
        return self.win.units


DEFAULT_LEFT = -1.0
DEFAULT_TOP = 1.0
DEFAULT_RIGHT = 1.0
DEFAULT_BOTTOM = -1.0


class Layout(Container):
    """Class with methods for positioning elements within a parent container.
    """

    def __init__(self,
                 parent: Container,
                 left: float = DEFAULT_LEFT,
                 top: float = DEFAULT_TOP,
                 right: float = DEFAULT_RIGHT,
                 bottom: float = DEFAULT_BOTTOM):

        self.parent = parent
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.check_invariants()

    def check_invariants(self):
        """Check that all invariants hold true."""
        # TODO: units could be configurable; min and max depends on units.
        # https://psychopy.org/general/units.html#units
        assert self.parent.units == "norm", "Position calculations assume norm units."

        assert (0.0 <= self.height <= 2.0), "Height must be in norm units."
        assert (0.0 <= self.width <= 2.0), "Width must be in norm units."
        assert (-1.0 <= self.top <= 1.0), "Top must be a y-value in norm units"
        assert (-1.0 <= self.left <=
                1.0), "Left must be an x-value in norm units"
        assert (-1.0 <= self.bottom <=
                1.0), "Bottom must be a y-value in norm units"
        assert (-1.0 <= self.right <=
                1.0), "Right must be an x-value in norm units"

        assert 0 < self.width <= self.parent.size[
            0], "Width must be greater than 0 and fit within the parent width."
        assert 0 < self.height <= self.parent.size[
            1], "Height must be greater than 0 and fit within the parent height."

    @property
    def win(self) -> visual.Window:
        """Returns the Window on which the layout will be drawn."""
        return self.parent.win

    @property
    def units(self) -> str:
        """Units for drawing; affects calculations related to position, size and radius."""
        return self.parent.units

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

    def above(self, y_coordinate: float, amount: float) -> float:
        """Returns a new y_coordinate value that is above the provided value
        by the given amount."""
        # assert self.bottom <= y_coordinate <= self.top, "y_coordinate out of range"
        assert amount >= 0, 'Amount must be positive'
        return y_coordinate + amount

    def below(self, y_coordinate: float, amount: float) -> float:
        """Returns a new y_coordinate value that is below the provided value
        by the given amount."""
        # assert self.bottom <= y_coordinate <= self.top, "y_coordinate out of range"
        assert amount >= 0, 'Amount must be positive'
        return y_coordinate - amount

    def right_of(self, x_coordinate: float, amount: float) -> float:
        """Returns a new x_coordinate value that is to the right of the
        provided value by the given amount."""
        # assert self.left <= x_coordinate <= self.right, "y_coordinate out of range"
        assert amount >= 0, 'Amount must be positive'
        return x_coordinate + amount

    def left_of(self, x_coordinate: float, amount: float) -> float:
        """Returns a new x_coordinate value that is to the left of the
        provided value by the given amount."""
        # assert self.left <= x_coordinate <= self.right, "y_coordinate out of range"
        assert amount >= 0, 'Amount must be positive'
        return x_coordinate - amount


class Alignment(Enum):
    CENTERED = 1
    LEFT = 2
    RIGHT = 3


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


def horizontally_aligned(parent: Layout,
                         alignment: Alignment,
                         width: float = None) -> Tuple[float, float]:
    """Returns a tuple of (left, right), indicating the values to be used"""

    left = parent.left
    right = parent.right
    if width:
        margin = parent.width - width
        if alignment == Alignment.CENTERED:
            half_margin = margin / 2
            left = left + half_margin
            right = right - half_margin
        elif alignment == Alignment.LEFT:
            right = right - margin
        elif alignment == Alignment.RIGHT:
            left = left + margin
    return left, right


def below(layout: Layout,
          width_pct: float = 1.0,
          alignment: Alignment = Alignment.CENTERED):
    """Constructs a layout immediately below another layout.

    Parameters
    ----------
        layout - another layout relative to the one constructed. Other
            properties will be copied from this one.
        width_pct - percentage of the available width.
    """
    width = layout.width * width_pct

    left, right = horizontally_aligned(layout, alignment, width)
    top = layout.bottom
    # bottom = top - height if height else DEFAULT_BOTTOM
    bottom = DEFAULT_BOTTOM

    return Layout(parent=layout.parent,
                  left=left,
                  top=top,
                  right=right,
                  bottom=bottom)