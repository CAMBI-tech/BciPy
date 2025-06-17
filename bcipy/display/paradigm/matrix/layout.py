"""Functions for calculating matrix layouts.

This module provides functionality for calculating and managing the layout of symbols
in a matrix-style grid format, commonly used in BCI paradigms. It handles the
positioning and spacing of elements based on window dimensions and layout requirements.
"""

from typing import List, Optional, Tuple

from bcipy.display.components.layout import (Layout, above, below, left_of,
                                             right_of, scaled_height,
                                             scaled_width)


def symbol_positions(
    container: Layout,
    rows: int,
    columns: int,
    symbol_set: List[str],
    max_spacing: Optional[float] = None
) -> List[Tuple[float, float]]:
    """Compute the positions for arranging a number of symbols in a grid layout.

    This function calculates the positions for placing symbols in a grid format,
    taking into account window dimensions, aspect ratio, and spacing requirements.
    The grid is centered in the container and positions are returned in row-major order.

    Args:
        container (Layout): Container in which the grid should be placed; must have a
            visual.Window parent, which is used to determine the aspect ratio.
        rows (int): Number of rows in the grid.
        columns (int): Number of columns in the grid.
        symbol_set (List[str]): List of symbols to place in the grid.
        max_spacing (Optional[float]): Optional max spacing (in layout units) in the height
            direction; width will be normalized to this value if provided. Defaults to None.

    Returns:
        List[Tuple[float, float]]: List of (x,y) tuples with (rows * columns) positions
            in row-major order.

    Raises:
        AssertionError: If container has no parent, if rows or columns are less than 1,
            or if there are not enough positions for all symbols.
    """
    assert container.parent, "Container must have a parent"
    assert rows >= 1 and columns >= 1, "There must be at least one row and one column"
    assert rows * columns >= len(symbol_set), \
        f"Not enough positions for symbols {len(symbol_set)}. Increase rows or columns."

    # compute the spacing (in container units) from the container width and height
    win_size = container.parent.size
    win_width, win_height = win_size

    horizontal_spacing = container.width / (columns + 1)
    vertical_spacing = container.height / (rows + 1)

    # determine which is smaller after scaling to the window aspect ratio
    if (win_width > win_height):
        # wider than tall
        scaled_horizontal_spacing = scaled_height(horizontal_spacing, win_size)
        scaled_vertical_spacing = vertical_spacing
    elif (win_height > win_width):
        # taller than wide
        scaled_horizontal_spacing = horizontal_spacing
        scaled_vertical_spacing = scaled_width(vertical_spacing, win_size)
    else:
        # square window
        scaled_horizontal_spacing = horizontal_spacing
        scaled_vertical_spacing = vertical_spacing

    if max_spacing and vertical_spacing > max_spacing:
        vertical_spacing = max_spacing
    # Use the minimum spacing
    if scaled_horizontal_spacing < scaled_vertical_spacing:
        vertical_spacing = scaled_height(horizontal_spacing, win_size)
    else:
        horizontal_spacing = scaled_width(vertical_spacing, win_size)

    # Work back from center to compute the starting position
    center_x, center_y = container.center
    spaces_left_of_center = int(columns / 2)
    start_pos_left = left_of(center_x,
                             spaces_left_of_center * horizontal_spacing)
    if columns % 2 == 0 and columns > 1:
        # even number of columns; adjust start_pos so that center_x is between
        # the middle two items.
        start_pos_left = right_of(start_pos_left, horizontal_spacing / 2)

    spaces_above_center = int(rows / 2)
    start_pos_top = above(center_y, spaces_above_center * vertical_spacing)
    if rows % 2 == 0 and rows > 1:
        start_pos_top = below(start_pos_top, vertical_spacing / 2)

    # adjust the beginning x,y values so adding a space results in the first
    # position.
    x_coord = left_of(start_pos_left, horizontal_spacing)
    y_coord = above(start_pos_top, vertical_spacing)

    positions = []
    for _row in range(rows):
        y_coord = below(y_coord, vertical_spacing)
        x_coord = left_of(start_pos_left, horizontal_spacing)
        for _col in range(columns):
            x_coord = right_of(x_coord, horizontal_spacing)
            positions.append((x_coord, y_coord))

    return positions
