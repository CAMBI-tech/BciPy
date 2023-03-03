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
        return visual.rect.Rect(win=self.layout.win,
                                units=self.layout.units,
                                lineWidth=2,
                                lineColor=self.config.task_color[0],
                                fillColor=None,
                                pos=self.layout.center,
                                size=(self.layout.width, self.layout.height))

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
            'text': self.config.task_text,
            'pos': self.layout.center,
            'units': self.layout.units,
            'font': self.config.task_font,
            'height': self.config.task_height,
            'color': self.config.task_color[0]
        }


class CalibrationTaskBar(TaskBar):
    """Task bar for Calibration tasks. Displays the count of inquiries.

    Parameters
    ----------
        win - visual.Window on which to render elements
        config - properties specifying fonts, colors, height, etc. The
            task_text property should be set to the inquiry count.
        current_index - index of the current inquiry
    """

    def __init__(self,
                 win: visual.Window,
                 config: TaskDisplayProperties = DEFAULT_TASK_PROPERTIES,
                 current_index: int = 1):
        self.inquiry_count = config.task_text
        self.current_index = current_index
        super().__init__(win, config)

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

        task = self.text_stim(text=self.config.task_text,
                              pos=self.layout.center,
                              anchorVert='bottom')

        spelled = self.text_stim(text=self.displayed_text(),
                                 pos=self.layout.center,
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
