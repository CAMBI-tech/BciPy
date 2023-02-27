"""Task bar component"""

from typing import Dict, Tuple
from psychopy import visual
from psychopy.visual.basevisual import BaseVisualStim


class TaskBar:
    """Component for displaying task-related information in a task window. The
    component elements are positioned at the top of the window.

    Parameters
    ----------
        win - visual.Window on which to render elements
        height - height in 'norm' units; window x-values range from -1 to +1,
            so the max height is 2.0. The default value is 0.2, which is 1/10
            of the screen height. See also:

            https://psychopy.org/general/units.html#units
    """
    def __init__(self, win: visual.Window, height: float = 0.2):
        assert win.units == "norm", "Position calculations assume norm units."
        assert (0.0 <= height <= 2.0), "Height must be in norm units."

        self.win = win
        self.units = 'norm'
        self.height = height
        self.stim = self.init_stim()

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""
        rect = visual.rect.Rect(win=self.win,
                                units=self.units,
                                lineWidth=2,
                                lineColor='white',
                                fillColor=None,
                                pos=self.center,
                                size=(self.width, self.height))  # (w, h)
        # task = self.centered_text_stim()
        task = self.left_aligned_text_stim()
        return {'task_text': task, 'rect': rect}

    def draw(self):
        """Draw the task bar elements."""
        for _key, stim in self.stim.items():
            stim.draw()

    def centered_text_stim(self) -> visual.TextStim:
        """Creates a centered text stim."""
        return visual.TextStim(
            win=self.win,
            units=self.units,
            pos=self.center,
            height=self.height / 2,  # TODO: what ratio makes sense?
            color='white',
            anchorVert='center',
            anchorHoriz='center')

    def left_aligned_text_stim(self) -> visual.TextStim:
        """Creates a left-aligned text stim
        TODO: consider padding
        """
        return visual.TextStim(
            win=self.win,
            units=self.units,
            pos=(self.left, self.vertical_middle),
            height=self.height / 2,  # TODO: what ratio makes sense?
            color='white',
            anchorHoriz='left')

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the component in norm units. Returns a (x,y) tuple."""
        # window ranges in both x and y from -1 to +1.
        return (self.horizontal_middle, self.vertical_middle)

    @property
    def horizontal_middle(self) -> float:
        """x-axis value in norm units for the midpoint of this component"""
        return 0.0

    @property
    def vertical_middle(self):
        """x-axis value in norm units for the midpoint of this component."""
        return self.top - (self.height / 2)

    @property
    def width(self) -> float:
        """Width in norm units of this component."""
        return self.right - self.left

    @property
    def top(self) -> float:
        """y-axis value in norm units for the highest point in this component."""
        return 1.0

    @property
    def bottom(self) -> float:
        """y-axis value in norm units for the lowest point for this component."""
        return 1.0 - self.height

    @property
    def left(self) -> float:
        """x-axis value in norm units for the left-most point."""
        return -1.0

    @property
    def right(self) -> float:
        """x-axis value in norm units for the right-most point."""
        return 1.0
