"""Task bar component"""

from typing import Dict, Tuple
from psychopy import visual
from psychopy.visual.basevisual import BaseVisualStim
import bcipy.display.components.layout as layout
from bcipy.display.main import TaskDisplayProperties

DEFAULT_TASK_PROPERTIES = TaskDisplayProperties(
    task_color=['white'],
    task_font='Courier New',
    task_pos=(0, 0.9),
    task_height=0.1,
    task_text='HELLO_WORLD\n           ')


class TaskBar:
    """Component for displaying task-related information in a task window. The
    component elements are positioned at the top of the window.

    Parameters
    ----------
        win - visual.Window on which to render elements
        config - properties specifying fonts, colors, height, etc.
    """
    def __init__(self,
                 win: visual.Window,
                 config: TaskDisplayProperties = DEFAULT_TASK_PROPERTIES):
        self.config = config
        self.layout = layout.at_top(layout.WindowContainer(win),
                                    self.compute_height())
        self.stim = self.init_stim()

    def compute_height(self):
        """Computes the component height using the provided config."""
        return (self.config.task_height * 2) + 0.05

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""

        task = self.centered_text_stim()
        # task = self.left_aligned_text_stim()
        return {'task_text': task, 'border': self.border_stim()}

    def draw(self):
        """Draw the task bar elements."""
        for _key, stim in self.stim.items():
            stim.draw()

    def border_stim(self):
        """Create the task bar outline"""
        return visual.rect.Rect(win=self.layout.win,
                                units=self.layout.units,
                                lineWidth=2,
                                lineColor=self.config.task_color,
                                fillColor=None,
                                pos=self.layout.center,
                                size=(self.layout.width, self.layout.height))

    def build_stim(self,
                   pos: Tuple[float, float],
                   vertical_alignment: str = 'center',
                   horizontal_alignment: str = 'center') -> visual.TextStim:
        """Builds a TextStim at the given position.

        Parameters
        ----------
            pos - stim position (x, y) coordinate, in norm units
            vertical_alignment - vertical alignment of 'top', 'bottom', or 'center'
            horizontal_alignment - horizontal alignment of 'left', 'right', or 'center'
        """
        assert vertical_alignment in ['top', 'bottom',
                                      'center'], "Invalid option"
        assert horizontal_alignment in ['left', 'right',
                                        'center'], "Invalid option"
        return visual.TextStim(name="task_text",
                               win=self.layout.win,
                               units=self.layout.units,
                               pos=pos,
                               text=self.config.task_text,
                               font=self.config.task_font,
                               height=self.config.task_height,
                               color=self.config.task_color[0],
                               anchorVert=vertical_alignment,
                               anchorHoriz=horizontal_alignment)

    def centered_text_stim(self) -> visual.TextStim:
        """Creates a centered text stim."""
        return self.build_stim(pos=self.layout.center,
                               vertical_alignment='center',
                               horizontal_alignment='center')

    def left_aligned_text_stim(self) -> visual.TextStim:
        """Creates a left-aligned text stim
        """
        return self.build_stim(pos=(self.layout.left,
                                    self.layout.vertical_middle),
                               vertical_alignment='center',
                               horizontal_alignment='left')
