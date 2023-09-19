"""Functionality for computing positions for elements within a VEP display"""

from itertools import cycle
from math import sqrt
from typing import List, NamedTuple, Optional, Tuple

from bcipy.display.components.layout import Layout


class CheckerboardSquare(NamedTuple):
    """Represents a single square within a checkerboard"""
    pos: Tuple[float, float]
    color: str
    size: Tuple[float, float]

    def inverse(self, colors: Tuple[str, str]) -> 'CheckerboardSquare':
        """Returns a new CheckerboardSquare whose color is the inverse
        of the current one.

        Parameters
        ----------
            colors - 2-tuple of color choices; the current square's color
                must be included.
        """
        assert len(colors) == 2, "Colors must be a 2-tuple"
        first, second = colors
        color = {first: second, second: first}[self.color]
        return CheckerboardSquare(self.pos, color, self.size)


def checkerboard(layout: Layout, squares: int, colors: Tuple[str, str],
                 center: Tuple[float, float],
                 board_size: Tuple[float, float]) -> List[CheckerboardSquare]:
    """Computes positions and colors for squares used to represent a
    checkerboard pattern. Returned positions are the center point of
    each square

    Parameters
    ----------
        layout - layout in which the checkerboard will be rendered. Used for
            positioning and scale.
        squares - total number of squares in the board; must be a perfect
            square (ex. 4, 9, 16, 25)
        colors - tuple of color names between which to alternate
        center - position of the center point of the board
        board_size - size in layout units of the entire checkerboard; square
            size will be computed from this.
    """
    boxes_per_row = int(sqrt(squares))
    assert boxes_per_row**2 == squares, "Must be a perfect square"

    square_width = board_size[0] / boxes_per_row
    square_height = board_size[1] / boxes_per_row
    square_size = (square_width, square_height)
    center_x, center_y = center

    # find the center position of the left_top square
    move_count = int(boxes_per_row / 2)
    if squares % 2 == 0:
        # with an even number the center is on a vertex; adjust by half a box
        move_count -= 0.5
    left = layout.left_of(center_x, square_width * move_count)
    top = layout.above(center_y, square_height * move_count)

    # iterate starting at the top left and proceeding in a zig zag
    # pattern to correspond with alternating checkerboard colors.
    positions = []
    for row in range(boxes_per_row):
        if row > 0:
            top = layout.below(top, square_height)
        for col in range(boxes_per_row):
            positions.append((left, top))
            if col < boxes_per_row - 1:
                if row % 2 == 0:
                    left = layout.right_of(left, square_width)
                else:
                    left = layout.left_of(left, square_width)
    return [
        CheckerboardSquare(*args)
        for args in zip(positions, cycle(colors), cycle([square_size]))
    ]


class BoxConfiguration():
    """Computes box size and positions for a VEP display.

    In this configuration, there is one row on top and one on the bottom.

    Parameters
    ----------
        num_boxes - number of boxes; currently supports 4 or 6.
        layout - defines the boundaries within a Window in which the text areas
            will be placed
        spacing_pct - used to specify spacing between boxes within a row.
        row_pct - what percentage of the height each row should use.
    """

    def __init__(self,
                 layout: Layout,
                 num_boxes: int,
                 spacing_pct: Optional[float] = None,
                 height_pct: float = 0.25):
        assert num_boxes == 4 or num_boxes == 6, 'Number of boxes must be 4 or 6'
        assert height_pct <= 0.5, "Rows can't take more than 50% of the height"
        self.num_boxes = num_boxes
        self.layout = layout

        default_spacing = {4: 0.1, 6: 0.05}
        if not spacing_pct:
            spacing_pct = default_spacing[num_boxes]
        self.spacing_pct = spacing_pct
        self.height_pct = height_pct
        self._row_count = 2

    def validate(self):
        """Validate invariants"""
        assert self.num_boxes == 4 or self.num_boxes == 6, 'Number of boxes must be 4 or 6'
        assert self.height_pct <= 0.5, "Rows can't take more than 50% of the height"

    def box_size(self, validate: bool = True) -> Tuple[float, float]:
        """Computes the size of each box"""
        if validate:
            self.validate()
        number_per_row = self.num_boxes / self._row_count

        # left and right boxes go to the edges, with a space between each box
        spaces_per_row = number_per_row - 1

        total_box_width_pct = 1 - (self.spacing_pct * spaces_per_row)
        width_pct = (total_box_width_pct / number_per_row)

        width = self.layout.width * width_pct
        height = self.layout.height * self.height_pct

        return (width, height)

    @property
    def units(self) -> str:
        """Position units"""
        return self.layout.units

    @property
    def positions(self) -> List[Tuple]:
        """Computes positions for text areas which contain symbols. Boxes are
        positioned in the corners for 4 elements. With 6 there are also areas
        in the middle.
        """
        self.validate()

        # size = box_size(num, layout, spacing_pct)
        width, height = self.box_size(validate=False)

        layout = self.layout
        top = layout.below(layout.top, (height / 2))
        bottom = layout.above(layout.bottom, (height / 2))

        left = layout.right_of(layout.left, width / 2)
        right = layout.left_of(layout.right, width / 2)

        positions = [(left, top), (right, top), (left, bottom),
                     (right, bottom)]
        if self.num_boxes == 6:
            positions += [(layout.horizontal_middle, top),
                          (layout.horizontal_middle, bottom)]
        return positions