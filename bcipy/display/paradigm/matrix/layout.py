"""Functions for calculating matrix layouts"""
from typing import List, Tuple

from bcipy.display.components.layout import Layout


def symbol_positions(container: Layout,
                     rows: int,
                     columns: int,
                     spacing: float = None) -> List[Tuple[float, float]]:
    """Compute the positions for arranging a number of symbols in a grid
    layout.

    Parameters
    ----------
        container - container in which the grid should be placed
        rows - number of rows in the grid
        columns - number of columns in the grid
        spacing - optional value to specify the space between positions
            (in the same units of the container)

    Returns
    -------
        list of (x,y) tuples with (rows * columns) positions in row,col order
    """
    assert rows >= 1 and columns >= 1, "There must be at least one row and one column"

    if not spacing:
        # compute the spacing (in container units) from the container width and height
        horizontal_spacing = container.width / (rows + 1)
        vertical_spacing = container.height / (columns + 1)
        spacing = min(horizontal_spacing, vertical_spacing)
    half_space = spacing / 2

    # Work back from center to compute the starting position
    center_x, center_y = container.center
    spaces_left_of_center = int(columns / 2)
    start_pos_left = container.left_of(center_x,
                                       spaces_left_of_center * spacing)
    if columns % 2 == 0 and columns > 1:
        # even number of columns; adjust start_pos so that center_x is between
        # the middle two items.
        start_pos_left = container.right_of(start_pos_left, half_space)

    spaces_above_center = int(rows / 2)
    start_pos_top = container.above(center_y, spaces_above_center * spacing)
    if rows % 2 == 0 and rows > 1:
        start_pos_top = container.below(start_pos_top, half_space)

    # adjust the beginning x,y values so adding a space results in the first
    # position.
    x_coord = container.left_of(start_pos_left, spacing)
    y_coord = container.above(start_pos_top, spacing)

    positions = []
    for _row in range(rows):
        y_coord = container.below(y_coord, spacing)
        x_coord = container.left_of(start_pos_left, spacing)
        for _col in range(columns):
            x_coord = container.right_of(x_coord, spacing)
            positions.append((x_coord, y_coord))

    return positions
