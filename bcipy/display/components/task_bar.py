"""Task bar component"""

from typing import Dict, Tuple
from psychopy import visual
from psychopy.visual.basevisual import BaseVisualStim
import bcipy.display.components.layout as layout
from bcipy.display.main import TaskDisplayProperties

DEFAULT_TASK_PROPERTIES = TaskDisplayProperties(task_color=['white'],
                                                task_font='Courier New',
                                                task_pos=(0, 0.9),
                                                task_height=0.1,
                                                task_text='')


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
        return self.config.task_height + 0.05

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""

        task = self.text_stim(text=self.config.task_text,
                              pos=(self.layout.left,
                                   self.layout.vertical_middle),
                              vertical_anchor='center',
                              horizontal_anchor='left')

        return {'task_text': task, 'border': self.border_stim()}

    def draw(self):
        """Draw the task bar elements."""
        for _key, stim in self.stim.items():
            stim.draw()

    def update(self, text: str = ''):
        """Update the task bar to display the given text."""
        self.stim['task_text'].text = text

    def border_stim(self):
        """Create the task bar outline"""
        return visual.rect.Rect(win=self.layout.win,
                                units=self.layout.units,
                                lineWidth=2,
                                lineColor=self.config.task_color,
                                fillColor=None,
                                pos=self.layout.center,
                                size=(self.layout.width, self.layout.height))

    def text_stim(self,
                  text: str,
                  pos: Tuple[float, float],
                  vertical_anchor: str = 'center',
                  horizontal_anchor: str = 'center') -> visual.TextStim:
        """Builds a TextStim at the given position using the configured properties.

        Parameters
        ----------
            text - content to display
            pos - stim position (x, y) coordinate, in norm units
            vertical_anchor - anchors text vertically at 'top', 'bottom', or 'center'.
            horizontal_anchor - anchors text horizontally at 'left', 'right', or 'center'.
        """
        assert vertical_anchor in ['top', 'bottom', 'center'], "Invalid option"
        assert horizontal_anchor in ['left', 'right',
                                     'center'], "Invalid option"
        return visual.TextStim(win=self.layout.win,
                               text=text,
                               pos=pos,
                               units=self.layout.units,
                               font=self.config.task_font,
                               height=self.config.task_height,
                               color=self.config.task_color[0],
                               anchorVert=vertical_anchor,
                               anchorHoriz=horizontal_anchor)


class CalibrationTaskBar(TaskBar):
    """Task bar for Calibration tasks. Displays the count of inquiries.

    Parameters
    ----------
        win - visual.Window on which to render elements
        config - properties specifying fonts, colors, height, etc.
        inquiry_count - total number of inquiries to complete
        current_index - index of the current inquiry
    """
    def __init__(self,
                 win: visual.Window,
                 config: TaskDisplayProperties = DEFAULT_TASK_PROPERTIES,
                 inquiry_count: int = 100,
                 current_index: int = 1):
        self.inquiry_count = inquiry_count
        self.current_index = current_index
        super().__init__(win, config)

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""

        task = self.text_stim(text=self.displayed_text(),
                              pos=(self.layout.left,
                                   self.layout.vertical_middle),
                              vertical_anchor='center',
                              horizontal_anchor='left')

        return {'task_text': task, 'border': self.border_stim()}

    def update(self, text: str = ''):
        """Update the displayed text"""
        if self.current_index < self.inquiry_count:
            self.current_index += 1
        self.stim['task_text'].text = self.displayed_text()

    def displayed_text(self) -> str:
        """Text to display. Computed from the current_index and inquiry_count. Ex. '(2/100)'"""
        return f"({self.current_index}/{self.inquiry_count})"


class CopyPhraseTaskBar(TaskBar):
    """Task bar for the Copy Phrase Task"""
    def __init__(self,
                 win: visual.Window,
                 config: TaskDisplayProperties = DEFAULT_TASK_PROPERTIES,
                 spelled_text: str = ''):

        # TODO: pad to length
        self.spelled_text = spelled_text
        super().__init__(win, config)

    def compute_height(self):
        """Computes the component height using the provided config."""
        return (self.config.task_height * 2) + 0.05

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""
        task = self.text_stim(text=self.config.task_text,
                              pos=self.layout.center,
                              vertical_anchor='bottom',
                              horizontal_anchor='center')
        spelled = self.text_stim(text=self.spelled_text,
                                 pos=self.layout.center,
                                 vertical_anchor='top',
                                 horizontal_anchor='center')
        spelled.setColor(self.config.task_color[-1])

        return {
            'task_text': task,
            'spelled_text': spelled,
            'border': self.border_stim()
        }

    def update(self, text: str = ''):
        """Update the task bar to display the given text."""
        # TODO: pad the text for alignment
        self.stim['spelled_text'].text = text
