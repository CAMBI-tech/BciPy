"""Task bar component"""

from typing import Dict, List
from psychopy import visual
from psychopy.visual.basevisual import BaseVisualStim
from psychopy.visual.line import Line
import bcipy.display.components.layout as layout


class TaskBar:
    """Component for displaying task-related information in a task window. The
    component elements are positioned at the top of the window.

    Parameters
    ----------
        win - visual.Window on which to render elements
        colors - Ordered list of colors to apply to task stimuli
        font - Font to apply to all task stimuli
        height - Height of all task text stimuli
        text - Task text to apply to stimuli
    """

    def __init__(self,
                 win: visual.Window,
                 colors: List[str] = None,
                 font: str = 'Courier New',
                 height: float = 0.1,
                 text: str = ''):
        self.colors = colors or ['white']
        self.font = font
        self.height = height
        self.text = text
        self.layout = layout.at_top(layout.WindowContainer(win),
                                    self.compute_height())
        self.stim = self.init_stim()

    def compute_height(self):
        """Computes the component height using the provided config."""
        return self.height + 0.05

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""
        task = self.text_stim()
        return {'task_text': task, 'border': self.border_stim()}

    def draw(self):
        """Draw the task bar elements."""
        for _key, stim in self.stim.items():
            stim.draw()

    def update(self, text: str = ''):
        """Update the task bar to display the given text."""
        self.stim['task_text'].text = text

    def border_stim(self) -> visual.rect.Rect:
        """Create the task bar outline"""
        return Line(win=self.layout.win,
                    units=self.layout.units,
                    start=(self.layout.left, self.layout.bottom),
                    end=(self.layout.right, self.layout.bottom),
                    lineColor=self.colors[0])

    def text_stim(self, **kwargs) -> visual.TextStim:
        """Constructs a TextStim. Uses the config to set default properties
        which may be overridden by providing keyword args.
        """

        props = {**self.default_text_props(), **kwargs}
        return visual.TextStim(**props)

    def default_text_props(self) -> dict:
        """Default properties for constructing a TextStim."""
        return {
            'win': self.layout.win,
            'text': self.text,
            'pos': self.layout.center,
            'units': self.layout.units,
            'font': self.font,
            'height': self.height,
            'color': self.colors[0]
        }


class CalibrationTaskBar(TaskBar):
    """Task bar for Calibration tasks. Displays the count of inquiries.

    Parameters
    ----------
        win - visual.Window on which to render elements
        inquiry_count - total number of inquiries to display
        current_index - index of the current inquiry
         **config - display config (colors, font, height)
    """

    def __init__(self,
                 win: visual.Window,
                 inquiry_count: int,
                 current_index: int = 1,
                 **config):
        self.inquiry_count = inquiry_count
        self.current_index = current_index
        super().__init__(win, **config)

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""

        task = self.text_stim(text=self.displayed_text(),
                              pos=self.layout.left_middle,
                              anchorHoriz='left',
                              alignText='left')

        return {'task_text': task, 'border': self.border_stim()}

    def update(self, text: str = ''):
        """Update the displayed text"""
        self.current_index += 1
        self.stim['task_text'].text = self.displayed_text()

    def displayed_text(self) -> str:
        """Text to display. Computed from the current_index and inquiry_count. Ex. '2/100'"""
        return f" {self.current_index}/{self.inquiry_count}"


class CopyPhraseTaskBar(TaskBar):
    """Task bar for the Copy Phrase Task

    Parameters
    ----------
        win - visual.Window on which to render elements
        task_text - text for the participant to spell
        spelled_text - text that has already been spelled
        **config - display config (colors, font, height)
    """

    def __init__(self,
                 win: visual.Window,
                 task_text: str = '',
                 spelled_text: str = '',
                 **config):
        self.task_text = task_text
        self.spelled_text = spelled_text
        super().__init__(win, **config)

    def compute_height(self):
        """Computes the component height using the provided config."""
        return (self.height * 2) + 0.05

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""

        task = self.text_stim(text=self.task_text,
                              pos=self.layout.center,
                              anchorVert='bottom')

        spelled = self.text_stim(text=self.displayed_text(),
                                 pos=self.layout.center,
                                 color=self.colors[-1],
                                 anchorVert='top')

        return {
            'task_text': task,
            'spelled_text': spelled,
            'border': self.border_stim()
        }

    def update(self, text: str = ''):
        """Update the task bar to display the given text."""
        self.spelled_text = text
        self.stim['spelled_text'].text = self.displayed_text()

    def displayed_text(self):
        """Spelled text padded for alignment."""
        diff = len(self.task_text) - len(self.spelled_text)
        if (diff > 0):
            return self.spelled_text + (' ' * diff)
        return self.spelled_text
