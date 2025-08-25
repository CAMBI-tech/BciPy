# mypy: disable-error-code="override"
"""Defines common functionality for GUI layouts.

This module provides classes and functions for managing layout and positioning
of GUI elements in the BciPy system. It includes utilities for alignment,
scaling, and positioning of components within containers.
"""
from enum import Enum
from typing import List, Optional, Protocol, Tuple


class Container(Protocol):
    """Protocol for an enclosing container with units and size.

    Attributes:
        size (Tuple[float, float]): Size of the container as (width, height).
        units (str): Units used for measurements (e.g., 'norm', 'height').
    """
    size: Tuple[float, float]
    units: str


# for norm units
DEFAULT_LEFT = -1.0
DEFAULT_TOP = 1.0
DEFAULT_RIGHT = 1.0
DEFAULT_BOTTOM = -1.0


class Alignment(Enum):
    """Specifies how elements should be aligned spatially.

    This enum defines the possible alignment options for positioning elements
    within a container.
    """
    CENTERED = 1
    LEFT = 2
    RIGHT = 3
    TOP = 4
    BOTTOM = 5

    @classmethod
    def horizontal(cls) -> List['Alignment']:
        """Get subset used for horizontal alignment.

        Returns:
            List[Alignment]: List of horizontal alignment options.
        """
        return [Alignment.CENTERED, Alignment.LEFT, Alignment.RIGHT]

    @classmethod
    def vertical(cls) -> List['Alignment']:
        """Get subset used for vertical alignment.

        Returns:
            List[Alignment]: List of vertical alignment options.
        """
        return [Alignment.CENTERED, Alignment.TOP, Alignment.BOTTOM]


# Positioning functions
def above(y_coordinate: float, amount: float) -> float:
    """Returns a new y_coordinate value that is above the provided value.

    Args:
        y_coordinate (float): Base y-coordinate.
        amount (float): Distance to move upward.

    Returns:
        float: New y-coordinate value.

    Raises:
        AssertionError: If amount is negative.
    """
    assert amount >= 0, 'Amount must be positive'
    return y_coordinate + amount


def below(y_coordinate: float, amount: float) -> float:
    """Returns a new y_coordinate value that is below the provided value.

    Args:
        y_coordinate (float): Base y-coordinate.
        amount (float): Distance to move downward.

    Returns:
        float: New y-coordinate value.

    Raises:
        AssertionError: If amount is negative.
    """
    assert amount >= 0, 'Amount must be positive'
    return y_coordinate - amount


def right_of(x_coordinate: float, amount: float) -> float:
    """Returns a new x_coordinate value that is to the right of the provided value.

    Args:
        x_coordinate (float): Base x-coordinate.
        amount (float): Distance to move right.

    Returns:
        float: New x-coordinate value.

    Raises:
        AssertionError: If amount is negative.
    """
    assert amount >= 0, 'Amount must be positive'
    return x_coordinate + amount


def left_of(x_coordinate: float, amount: float) -> float:
    """Returns a new x_coordinate value that is to the left of the provided value.

    Args:
        x_coordinate (float): Base x-coordinate.
        amount (float): Distance to move left.

    Returns:
        float: New x-coordinate value.

    Raises:
        AssertionError: If amount is negative.
    """
    assert amount >= 0, 'Amount must be positive'
    return x_coordinate - amount


def envelope(pos: Tuple[float, float],
             size: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Compute the vertices for the envelope of a shape.

    Args:
        pos (Tuple[float, float]): Center position of the shape.
        size (Tuple[float, float]): Size of the shape as (width, height).

    Returns:
        List[Tuple[float, float]]: List of vertices defining the shape's envelope.
    """
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
    """Scales the provided height value to reflect the aspect ratio of a window.

    Args:
        height (float): Height value to scale.
        window_size (Tuple[float, float]): Window dimensions as (width, height).
        units (str): Units to use for scaling. Defaults to 'norm'.

    Returns:
        Tuple[float, float]: Scaled size as (width, height).
    """
    if units == 'height':
        width = height
        return (width, height)

    win_width, win_height = window_size
    width = (win_height / win_width) * height
    return (width, height)


def scaled_height(width: float,
                  window_size: Tuple[float, float],
                  units: str = 'norm') -> float:
    """Given a width, find the equivalent height scaled to the aspect ratio.

    Args:
        width (float): Width value to scale.
        window_size (Tuple[float, float]): Window dimensions as (width, height).
        units (str): Units to use for scaling. Defaults to 'norm'.

    Returns:
        float: Scaled height value.
    """
    if units == 'height':
        return width
    win_width, win_height = window_size
    return width / (win_height / win_width)


def scaled_width(height: float,
                 window_size: Tuple[float, float],
                 units: str = 'norm') -> float:
    """Given a height, find the equivalent width scaled to the aspect ratio.

    Args:
        height (float): Height value to scale.
        window_size (Tuple[float, float]): Window dimensions as (width, height).
        units (str): Units to use for scaling. Defaults to 'norm'.

    Returns:
        float: Scaled width value.
    """
    width, _height = scaled_size(height, window_size, units)
    return width


class Layout(Container):
    """Class with methods for positioning elements within a parent container.

    This class provides functionality for managing the layout and positioning
    of GUI elements within a container, including methods for resizing and
    alignment.

    Attributes:
        units (str): Units used for measurements (e.g., 'norm', 'height').
        parent (Optional[Container]): Parent container if any.
        top (float): Top boundary position.
        left (float): Left boundary position.
        bottom (float): Bottom boundary position.
        right (float): Right boundary position.
    """

    def __init__(
        self,
        parent: Optional[Container] = None,
        left: float = DEFAULT_LEFT,
        top: float = DEFAULT_TOP,
        right: float = DEFAULT_RIGHT,
        bottom: float = DEFAULT_BOTTOM,
        units: str = "norm"
    ) -> None:
        """Initialize the Layout.

        Args:
            parent (Optional[Container]): Parent container. Defaults to None.
            left (float): Left boundary position. Defaults to DEFAULT_LEFT.
            top (float): Top boundary position. Defaults to DEFAULT_TOP.
            right (float): Right boundary position. Defaults to DEFAULT_RIGHT.
            bottom (float): Bottom boundary position. Defaults to DEFAULT_BOTTOM.
            units (str): Units to use. Defaults to "norm".
        """
        self.units: str = units
        self.parent = parent
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.check_invariants()

    def check_invariants(self) -> None:
        """Check that all invariants hold true.

        Raises:
            AssertionError: If any invariant is violated.
        """
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
        """Returns the (w,h) value scaled to reflect the aspect ratio.

        Args:
            height (float): Height value to scale.

        Returns:
            Tuple[float, float]: Scaled size as (width, height).

        Raises:
            AssertionError: If parent is not configured.
        """
        if self.units == 'height':
            width = height
            return (width, height)
        assert self.parent is not None, 'Parent must be configured'
        return scaled_size(height, self.parent.size, self.units)

    @property
    def size(self) -> Tuple[float, float]:
        """Get the layout size.

        Returns:
            Tuple[float, float]: Size as (width, height).
        """
        return (self.width, self.height)

    @property
    def width(self) -> float:
        """Get the width in norm units of this component.

        Returns:
            float: Width value.
        """
        return self.right - self.left

    @property
    def height(self) -> float:
        """Get the height in norm units of this component.

        Returns:
            float: Height value.
        """
        return self.top - self.bottom

    @property
    def left_top(self) -> Tuple[float, float]:
        """Get the top left position.

        Returns:
            Tuple[float, float]: Position as (x, y).
        """
        return (self.left, self.top)

    @property
    def right_bottom(self) -> Tuple[float, float]:
        """Get the bottom right position.

        Returns:
            Tuple[float, float]: Position as (x, y).
        """
        return (self.right, self.bottom)

    @property
    def horizontal_middle(self) -> float:
        """Get the x-axis value for the midpoint of this component.

        Returns:
            float: X-coordinate of the midpoint.
        """
        return (self.left + self.right) / 2

    @property
    def vertical_middle(self) -> float:
        """Get the y-axis value for the midpoint of this component.

        Returns:
            float: Y-coordinate of the midpoint.
        """
        return (self.top + self.bottom) / 2

    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the component.

        Returns:
            Tuple[float, float]: Center position as (x, y).
        """
        return (self.horizontal_middle, self.vertical_middle)

    @property
    def left_middle(self) -> Tuple[float, float]:
        """Get the point centered on the left-most edge.

        Returns:
            Tuple[float, float]: Position as (x, y).
        """
        return (self.left, self.vertical_middle)

    @property
    def right_middle(self) -> Tuple[float, float]:
        """Get the point centered on the right-most edge.

        Returns:
            Tuple[float, float]: Position as (x, y).
        """
        return (self.right, self.vertical_middle)

    def resize_width(self,
                     width_pct: float,
                     alignment: Alignment = Alignment.CENTERED) -> None:
        """Adjust the width of the current layout.

        Args:
            width_pct (float): Percentage of the current width.
            alignment (Alignment): Specifies how the remaining width should be aligned.
                Defaults to Alignment.CENTERED.

        Raises:
            AssertionError: If width_pct is not positive or alignment is invalid.
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
                      alignment: Alignment = Alignment.CENTERED) -> None:
        """Adjust the height of the current layout.

        Args:
            height_pct (float): Percentage of the current height.
            alignment (Alignment): Specifies how the remaining height should be aligned.
                Defaults to Alignment.CENTERED.

        Raises:
            AssertionError: If height_pct is not positive or alignment is invalid.
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
    """Constructs a layout of a given height that spans the full width of the window.

    Args:
        parent (Container): Parent container.
        height (float): Height value in 'norm' units.

    Returns:
        Layout: New layout instance positioned at the top.
    """
    top = DEFAULT_TOP
    return Layout(parent=parent,
                  left=DEFAULT_LEFT,
                  top=top,
                  right=DEFAULT_RIGHT,
                  bottom=top - height)


def at_bottom(parent: Container, height: float) -> Layout:
    """Constructs a layout of a given height that spans the full width of the window.

    Args:
        parent (Container): Parent container.
        height (float): Height value in 'norm' units.

    Returns:
        Layout: New layout instance positioned at the bottom.
    """
    bottom = DEFAULT_BOTTOM
    return Layout(parent=parent,
                  left=DEFAULT_LEFT,
                  top=DEFAULT_BOTTOM + height,
                  right=DEFAULT_RIGHT,
                  bottom=bottom)


def centered(parent: Optional[Container] = None,
             width_pct: float = 1.0,
             height_pct: float = 1.0) -> Layout:
    """Constructs a layout that is centered on the screen.

    Args:
        parent (Optional[Container]): Optional parent container.
        width_pct (float): Optional; sets the width to a given percentage of fullscreen.
            Defaults to 1.0.
        height_pct (float): Optional; sets the height to a given percentage of fullscreen.
            Defaults to 1.0.

    Returns:
        Layout: New centered layout instance.
    """
    container = Layout(parent=parent)
    container.resize_width(width_pct, alignment=Alignment.CENTERED)
    container.resize_height(height_pct, alignment=Alignment.CENTERED)
    return container


def from_envelope(verts: List[Tuple[float, float]]) -> Layout:
    """Constructs a layout from a list of vertices which comprise a shape's envelope.

    Args:
        verts (List[Tuple[float, float]]): List of vertices defining the shape's envelope.

    Returns:
        Layout: New layout instance based on the envelope.
    """
    x_coords, y_coords = zip(*verts)
    return Layout(left=min(x_coords),
                  top=max(y_coords),
                  right=max(x_coords),
                  bottom=min(y_coords))


def height_units(window_size: Tuple[float, float]) -> Layout:
    """Constructs a layout with height units using the given Window dimensions.

    Args:
        window_size (Tuple[float, float]): Window dimensions as (width, height).

    Returns:
        Layout: New layout instance using height units.

    Note:
        For an aspect ratio of 4:3:
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
