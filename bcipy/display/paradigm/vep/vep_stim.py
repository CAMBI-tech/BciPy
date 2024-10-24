"""Display components for VEP"""
from typing import List, Tuple

from psychopy import visual  # type: ignore
from psychopy.visual.shape import ShapeStim  # type: ignore

from bcipy.display.components.layout import envelope
from bcipy.display.paradigm.vep.layout import checkerboard


class VEPStim:
    """Represents a checkerboard of squares that can be flashed at a given
    rate. Flashing is accomplished by inverting the colors of each square.

    Parameters
    ----------
        layout - used to build the stimulus
        code - A list of integers representing the VEP code for each box
        colors - tuple of colors for the checkerboard pattern
        center - center position of the checkerboard
        size - size of the checkerboard, in layout units
        num_squares - number of squares in the checkerboard
    """

    def __init__(self,
                 win: visual.Window,
                 code: List[int],
                 colors: Tuple[str, str],
                 center: Tuple[float, float],
                 size: Tuple[float, float],
                 num_squares: int = 4):
        self.window = win
        self.code = code
        self.colors = colors

        squares = checkerboard(squares=num_squares,
                               colors=colors,
                               center=center,
                               board_size=size)
        board_boundary = envelope(pos=center, size=size)
        self.bounds = board_boundary

        frame1_holes = []
        frame2_holes = []
        for square in squares:
            square_boundary = envelope(pos=square.pos, size=square.size)
            # squares define the holes in the polygon
            if square.color == colors[0]:
                frame2_holes.append(square_boundary)
            elif square.color == colors[1]:
                frame1_holes.append(square_boundary)

        # Checkerboard is represented as a polygon with holes, backed by a
        # simple square with the alternating color.
        # This technique renders more efficiently and scales better than using
        # separate shapes (Rect or Gradient) for each square.
        background = ShapeStim(self.window,
                               lineColor=colors[1],
                               fillColor=colors[1],
                               vertices=board_boundary)
        self.on_stim = [
            background,
            # polygon with holes
            ShapeStim(self.window,
                      lineWidth=0,
                      fillColor=colors[0],
                      closeShape=True,
                      vertices=[board_boundary, *frame1_holes])
        ]
        self.off_stim = [
            background,
            # polygon with holes
            ShapeStim(self.window,
                      lineWidth=0,
                      fillColor=colors[0],
                      closeShape=True,
                      vertices=[board_boundary, *frame2_holes])
        ]

    def render_frame(self, frame: int) -> None:
        """Render a given frame number, where frame refers to a code index"""
        if self.code[frame] == 1:
            self.frame_on()
        else:
            self.frame_off()

    def frame_on(self) -> None:
        """Each square is set to a starting color and draw."""
        for stim in self.on_stim:
            stim.draw()

    def frame_off(self) -> None:
        """Invert each square from its starting color and draw."""
        for stim in self.off_stim:
            stim.draw()
