"""Task bar component.

This module provides components for displaying task-related information in a task window.
It includes base task bar functionality and specialized implementations for different
types of tasks like calibration and copy phrase tasks.
"""

from typing import Dict, List, Optional, Any

from psychopy import visual
from psychopy.visual.basevisual import BaseVisualStim

import bcipy.display.components.layout as layout


class TaskBar:
    """Component for displaying task-related information in a task window.

    The component elements are positioned at the top of the window.

    Attributes:
        win (visual.Window): Window on which to render elements.
        colors (List[str]): Ordered list of colors to apply to task stimuli.
        font (str): Font to apply to all task stimuli.
        height (float): Height of all task text stimuli.
        padding (float): Padding used in conjunction with text height to compute
            the overall height of the task bar.
        text (str): Task text to apply to stimuli.
        layout (layout.Layout): Layout manager for positioning elements.
        stim (Dict[str, BaseVisualStim]): Dictionary of visual stimuli.
    """

    def __init__(
        self,
        win: visual.Window,
        colors: Optional[List[str]] = None,
        font: str = 'Courier New',
        height: float = 0.1,
        text: str = '',
        padding: Optional[float] = None
    ) -> None:
        """Initialize the TaskBar.

        Args:
            win (visual.Window): Window on which to render elements.
            colors (Optional[List[str]]): Ordered list of colors to apply to task stimuli.
                Defaults to ['white'].
            font (str): Font to apply to all task stimuli. Defaults to 'Courier New'.
            height (float): Height of all task text stimuli. Defaults to 0.1.
            text (str): Task text to apply to stimuli. Defaults to ''.
            padding (Optional[float]): Padding used in conjunction with text height.
                If None, defaults to height/2.
        """
        self.win = win
        self.colors = colors or ['white']
        self.font = font
        self.height = height
        self.padding = (height / 2) if padding is None else padding
        self.text = text
        self.layout = layout.at_top(win, self.compute_height())
        self.stim = self.init_stim()

    @property
    def height_pct(self) -> float:
        """Get the percentage of the total window that the task bar occupies.

        Returns:
            float: Percentage value between 0 and 1.
        """
        win_layout = layout.Layout(self.win)
        return self.compute_height() / win_layout.height

    def compute_height(self) -> float:
        """Compute the component height using the provided config.

        Returns:
            float: Total height of the task bar.
        """
        return self.height + self.padding

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements.

        Returns:
            Dict[str, BaseVisualStim]: Dictionary of initialized visual stimuli.
        """
        task = self.text_stim()
        return {'task_text': task, 'border': self.border_stim()}

    def draw(self) -> None:
        """Draw the task bar elements."""
        for _key, stim in self.stim.items():
            stim.draw()

    def update(self, text: str = '') -> None:
        """Update the task bar to display the given text.

        Args:
            text (str): New text to display. Defaults to ''.
        """
        self.stim['task_text'].text = text

    def border_stim(self) -> visual.Line:
        """Create the task bar outline.

        Returns:
            visual.Line: Line stimulus representing the task bar border.
        """
        # pylint: disable=not-callable
        return visual.Line(
            win=self.win,
            units=self.layout.units,
            start=(self.layout.left, self.layout.bottom),
            end=(self.layout.right, self.layout.bottom),
            lineColor=self.colors[0])

    def text_stim(self, **kwargs) -> visual.TextStim:
        """Construct a TextStim with default properties.

        Uses the config to set default properties which may be overridden by
        providing keyword args.

        Args:
            **kwargs: Additional properties to override defaults.

        Returns:
            visual.TextStim: Configured text stimulus.
        """
        props = {**self.default_text_props(), **kwargs}
        return visual.TextStim(**props)

    def default_text_props(self) -> Dict[str, Any]:
        """Get default properties for constructing a TextStim.

        Returns:
            Dict[str, Any]: Dictionary of default text stimulus properties.
        """
        return {
            'win': self.win,
            'text': self.text,
            'pos': self.layout.center,
            'units': self.layout.units,
            'font': self.font,
            'height': self.height,
            'color': self.colors[0]
        }


class CalibrationTaskBar(TaskBar):
    """Task bar for Calibration tasks.

    Displays the count of inquiries in the format "current/total".

    Attributes:
        inquiry_count (int): Total number of inquiries to display.
        current_index (int): Index of the current inquiry.
    """

    def __init__(
        self,
        win: visual.Window,
        inquiry_count: int,
        current_index: int = 1,
        **config: Any
    ) -> None:
        """Initialize the CalibrationTaskBar.

        Args:
            win (visual.Window): Window on which to render elements.
            inquiry_count (int): Total number of inquiries to display.
            current_index (int): Index of the current inquiry. Defaults to 1.
            **config: Additional display configuration (colors, font, height).
        """
        self.inquiry_count = inquiry_count
        self.current_index = current_index
        super().__init__(win, **config)

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements.

        Returns:
            Dict[str, BaseVisualStim]: Dictionary of initialized visual stimuli.
        """
        task = self.text_stim(
            text=self.displayed_text(),
            pos=self.layout.left_middle,
            anchorHoriz='left',
            alignText='left'
        )

        return {'task_text': task, 'border': self.border_stim()}

    def update(self, text: str = '') -> None:
        """Update the displayed text.

        Args:
            text (str): Unused parameter. Defaults to ''.
        """
        self.current_index += 1
        self.stim['task_text'].text = self.displayed_text()

    def displayed_text(self) -> str:
        """Get the text to display.

        Returns:
            str: Text in the format "current/total" (e.g., "2/100").
        """
        return f" {self.current_index}/{self.inquiry_count}"


class CopyPhraseTaskBar(TaskBar):
    """Task bar for the Copy Phrase Task.

    Displays both the target text and the currently spelled text.

    Attributes:
        task_text (str): Text for the participant to spell.
        spelled_text (str): Text that has already been spelled.
    """

    def __init__(
        self,
        win: visual.Window,
        task_text: str = '',
        spelled_text: str = '',
        **config: Any
    ) -> None:
        """Initialize the CopyPhraseTaskBar.

        Args:
            win (visual.Window): Window on which to render elements.
            task_text (str): Text for the participant to spell. Defaults to ''.
            spelled_text (str): Text that has already been spelled. Defaults to ''.
            **config: Additional display configuration (colors, font, height).
        """
        self.task_text = task_text
        self.spelled_text = spelled_text
        super().__init__(win, **config)

    def compute_height(self) -> float:
        """Compute the component height using the provided config.

        Height is doubled to account for task_text and spelled_text being on
        separate lines.

        Returns:
            float: Total height of the task bar.
        """
        return (self.height * 2) + self.padding

    def init_stim(self) -> Dict[str, BaseVisualStim]:
        """Initialize the stimuli elements.

        Returns:
            Dict[str, BaseVisualStim]: Dictionary of initialized visual stimuli.
        """
        task = self.text_stim(
            text=self.task_text,
            pos=self.layout.center,
            anchorVert='bottom'
        )

        spelled = self.text_stim(
            text=self.displayed_text(),
            pos=self.layout.center,
            color=self.colors[-1],
            anchorVert='top'
        )

        return {
            'task_text': task,
            'spelled_text': spelled,
            'border': self.border_stim()
        }

    def update(self, text: str = '') -> None:
        """Update the task bar to display the given text.

        Args:
            text (str): New spelled text to display. Defaults to ''.
        """
        self.spelled_text = text
        self.stim['spelled_text'].text = self.displayed_text()

    def displayed_text(self) -> str:
        """Get the spelled text padded for alignment.

        Returns:
            str: Spelled text padded with spaces to match task_text length.
        """
        diff = len(self.task_text) - len(self.spelled_text)
        if (diff > 0):
            return self.spelled_text + (' ' * diff)
        return self.spelled_text
