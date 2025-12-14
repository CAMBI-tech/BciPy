"""RSVP copy phrase display module.

This module provides the RSVP copy phrase display implementation which handles the visual
presentation of stimuli during copy phrase tasks. It extends the base RSVP display with
copy phrase-specific functionality.

Note:
    RSVP Tasks are RSVPDisplay objects with different structure. They share
    the tasks and the essential elements and stimuli. However, layout, length of
    stimuli list, update procedures and colors are different. Therefore each
    mode should be separated from each other carefully.
"""

from typing import Optional

from psychopy import core, visual

from bcipy.core.stimuli import resize_image
from bcipy.core.symbols import SPACE_CHAR
from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.rsvp.display import BCIPY_LOGO_PATH, RSVPDisplay


class CopyPhraseDisplay(RSVPDisplay):
    """Copy Phrase display object of RSVP.

    This class extends the RSVPDisplay to provide copy phrase-specific functionality.
    It handles the visual presentation of stimuli during copy phrase tasks, including
    preview functionality and wait screen display.

    Attributes:
        starting_spelled_text (str): Initial text that has been spelled.
        static_task_text (str): Target text for the user to attempt to spell.
        static_task_color (str): Target text color for the user to attempt to spell.
    """

    def __init__(
            self,
            window: visual.Window,
            clock: core.Clock,
            experiment_clock: core.Clock,
            stimuli: StimuliProperties,
            task_bar: TaskBar,
            info: InformationProperties,
            starting_spelled_text: str = '',
            trigger_type: str = 'image',
            space_char: str = SPACE_CHAR,
            preview_config: Optional[PreviewParams] = None,
            full_screen: bool = False) -> None:
        """Initialize Copy Phrase Task Objects.

        Args:
            window (visual.Window): PsychoPy window for display.
            clock (core.Clock): Clock for timing.
            experiment_clock (core.Clock): Clock for experiment timing.
            stimuli (StimuliProperties): Properties for stimulus presentation.
            task_bar (TaskBar): Task bar component for progress display.
            info (InformationProperties): Properties for information display.
            starting_spelled_text (str, optional): Initial text that has been spelled.
                Defaults to ''.
            trigger_type (str, optional): Type of trigger to use. Defaults to 'image'.
            space_char (str, optional): Character to use for spaces. Defaults to SPACE_CHAR.
            preview_config (Optional[PreviewParams], optional): Configuration for preview
                functionality. Defaults to None.
            full_screen (bool, optional): Whether to display in fullscreen mode.
                Defaults to False.
        """
        self.starting_spelled_text = starting_spelled_text

        super().__init__(window,
                         clock,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         trigger_type=trigger_type,
                         space_char=space_char,
                         preview_config=preview_config,
                         full_screen=full_screen)

    @property
    def preview_index(self) -> int:
        """Index within an inquiry at which the inquiry preview should be displayed.

        For copy phrase there is no target prompt so it should display before
        the fixation.

        Returns:
            int: The index at which to display the preview (0 for copy phrase).
        """
        return 0

    def wait_screen(self, message: str, message_color: str) -> None:
        """Display a wait screen with a message and optional logo.

        Args:
            message (str): Message to be displayed while waiting.
            message_color (str): Color of the message to be displayed.

        Raises:
            Exception: If the logo image cannot be loaded.
        """
        self.draw_static()

        # Construct the wait message
        wait_message = visual.TextStim(win=self.window,
                                       font=self.stimuli_font,
                                       text=message,
                                       height=.1,
                                       color=message_color,
                                       pos=(0, -.55),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1,
                                       depth=-6.0)

        # try adding the BciPy logo to the wait screen
        try:
            wait_logo = visual.ImageStim(
                self.window,
                image=BCIPY_LOGO_PATH,
                pos=(0, 0),
                mask=None,
                ori=0.0)
            wait_logo.size = resize_image(
                BCIPY_LOGO_PATH,
                self.window.size,
                1)
            wait_logo.draw()

        except Exception as e:
            self.logger.exception(
                f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            raise e

        # Draw and flip the screen.
        wait_message.draw()
        self.window.flip()
