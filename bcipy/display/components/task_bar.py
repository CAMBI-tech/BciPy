"""Task bar component"""

from typing import Dict
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
        task = visual.TextStim(win=self.layout.win,
                               text=self.config.task_text,
                               pos=self.layout.left_middle,
                               units=self.layout.units,
                               font=self.config.task_font,
                               height=self.config.task_height,
                               color=self.config.task_color[0],
                               anchorVert='center',
                               anchorHoriz='left',
                               alignText='left')
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
        return visual.rect.Rect(win=self.layout.win,
                                units=self.layout.units,
                                lineWidth=2,
                                lineColor=self.config.task_color,
                                fillColor=None,
                                pos=self.layout.center,
                                size=(self.layout.width, self.layout.height))


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

        task = visual.TextStim(win=self.layout.win,
                               text=self.displayed_text(),
                               pos=self.layout.left_middle,
                               units=self.layout.units,
                               font=self.config.task_font,
                               height=self.config.task_height,
                               color=self.config.task_color[0],
                               anchorHoriz='left',
                               alignText='left')

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

        self.spelled_text = spelled_text
        super().__init__(win, config)

    def compute_height(self):
        """Computes the component height using the provided config."""
        return (self.config.task_height * 2) + 0.05

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements."""

        task = visual.TextStim(win=self.layout.win,
                               text=self.config.task_text,
                               pos=self.layout.center,
                               units=self.layout.units,
                               font=self.config.task_font,
                               height=self.config.task_height,
                               color=self.config.task_color[0],
                               anchorVert='bottom')

        spelled = visual.TextStim(win=self.layout.win,
                                  text=self.displayed_text(),
                                  pos=self.layout.center,
                                  units=self.layout.units,
                                  font=self.config.task_font,
                                  height=self.config.task_height,
                                  color=self.config.task_color[-1],
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
        diff = len(self.config.task_text) - len(self.spelled_text)
        if (diff > 0):
            return self.spelled_text + (' ' * diff)
        return self.spelled_text