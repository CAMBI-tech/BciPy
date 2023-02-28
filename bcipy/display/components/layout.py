"""Defines common functionality for GUI layouts."""
from typing import Tuple
from psychopy import visual


class Container:
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
    def center(self) -> Tuple[float, float]:
        """Center point of the component in norm units. Returns a (x,y) tuple."""
        return (self.horizontal_middle, self.vertical_middle)

    @property
    def horizontal_middle(self) -> float:
        """x-axis value in norm units for the midpoint of this component"""
        return (self.left + self.right) / 2

    @property
    def vertical_middle(self) -> float:
        """x-axis value in norm units for the midpoint of this component."""
        return (self.top + self.bottom) / 2
