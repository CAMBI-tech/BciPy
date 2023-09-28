"""Defines common functionality for GUI layouts."""
from enum import Enum
from typing import Protocol, Tuple


class Container(Protocol):
    """Protocol for an enclosing container with units and size."""
    size: Tuple[float, float]
    units: str


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


class Layout(Container):
    """Class with methods for positioning elements within a parent container.
    """

    def __init__(self,
                 parent: Container = None,
                 left: float = DEFAULT_LEFT,
                 top: float = DEFAULT_TOP,
                 right: float = DEFAULT_RIGHT,
                 bottom: float = DEFAULT_BOTTOM):
        self.units = "norm"
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
        assert self.units == "norm", "Position calculations assume norm units."

        assert (0.0 <= self.height <= 2.0), "Height must be in norm units."
        assert (0.0 <= self.width <= 2.0), "Width must be in norm units."
        assert (-1.0 <= self.top <= 1.0), "Top must be a y-value in norm units"
        assert (-1.0 <= self.left <=
                1.0), "Left must be an x-value in norm units"
        assert (-1.0 <= self.bottom <=
                1.0), "Bottom must be a y-value in norm units"
        assert (-1.0 <= self.right <=
                1.0), "Right must be an x-value in norm units"

        if self.parent:
            assert 0 < self.width <= self.parent.size[
                0], "Width must be greater than 0 and fit within the parent width."
            assert 0 < self.height <= self.parent.size[
                1], "Height must be greater than 0 and fit within the parent height."

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


def centered(width_pct: float = 1.0, height_pct: float = 1.0) -> Layout:
    """Constructs a layout that is centered on the screen. Default size is
    fullscreen but optional parameters can be used to adjust the width and
    height.

    Parameters
    ----------
        width_pct - optional; sets the width to a given percentage of
          fullscreen.
        height_pct - optional; sets the height to a given percentage of
          fullscreen.
    """
    container = Layout()
    container.resize_width(width_pct, alignment=Alignment.CENTERED)
    container.resize_height(height_pct, alignment=Alignment.CENTERED)
    return container
