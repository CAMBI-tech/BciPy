"""Defines common functionality for GUI layouts."""
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


class Layout(Container):
    """Base class for a component with methods for positioning elements.
    """

    def __init__(self, parent: Container, left: float, top: float,
                 right: float, bottom: float):

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


def at_top(parent: Container, height: float):
    """Constructs a layout of a given height that spans the full width of the
    window and is positioned at the top.

    Parameters
    ----------
        height - value in 'norm' units
    """
    return Layout(parent=parent,
                  left=-1.0,
                  top=1.0,
                  right=1.0,
                  bottom=1.0 - height)


def centered(parent: Container, width: float = None, height: float = None):
    """Constructs a centered layout"""


def below(layout: Layout, width: float = None, height: float = None):
    """Constructs a layout below another layout.

    Parameters
    ----------
        layout - another layout relative to the one constructed. Other
            properties will be copied from this one.
    """
    # TODO: account for height
    return Layout(parent=layout.parent,
                  left=-1.0,
                  top=layout.bottom,
                  right=1.0,
                  bottom=-1.0)