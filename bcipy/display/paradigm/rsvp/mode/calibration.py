"""RSVP calibration display module.

This module provides the RSVP calibration display implementation which handles the visual
presentation of stimuli during calibration tasks. It extends the base RSVP display with
calibration-specific functionality.
"""
from typing import Optional

from psychopy import core, visual

from bcipy.core.symbols import SPACE_CHAR
from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.rsvp.display import RSVPDisplay


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display for RSVP paradigm.

    This class extends the RSVPDisplay to provide calibration-specific functionality.
    It handles the visual presentation of stimuli during calibration tasks, including
    preview functionality and timing control.

    Attributes:
        preview_index (int): Index within an inquiry at which the inquiry preview
            should be displayed. For calibration, this is set to 1 (after target prompt).
    """

    def __init__(self,
                 window: visual.Window,
                 static_clock: core.StaticPeriod,
                 experiment_clock: core.Clock,
                 stimuli: StimuliProperties,
                 task_bar: TaskBar,
                 info: InformationProperties,
                 trigger_type: str = 'image',
                 preview_config: Optional[PreviewParams] = None,
                 space_char: str = SPACE_CHAR,
                 full_screen: bool = False) -> None:
        """Initialize the RSVP calibration display.

        Args:
            window (visual.Window): PsychoPy window for display.
            static_clock (core.StaticPeriod): Clock for static timing.
            experiment_clock (core.Clock): Clock for experiment timing.
            stimuli (StimuliProperties): Properties for stimulus presentation.
            task_bar (TaskBar): Task bar component for progress display.
            info (InformationProperties): Properties for information display.
            trigger_type (str, optional): Type of trigger to use. Defaults to 'image'.
            preview_config (Optional[PreviewParams], optional): Configuration for preview
                functionality. Defaults to None.
            space_char (str, optional): Character to use for spaces. Defaults to SPACE_CHAR.
            full_screen (bool, optional): Whether to display in fullscreen mode.
                Defaults to False.
        """
        super().__init__(window,
                         static_clock,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         trigger_type=trigger_type,
                         preview_config=preview_config,
                         space_char=space_char,
                         full_screen=full_screen)

    @property
    def preview_index(self) -> int:
        """Index within an inquiry at which the inquiry preview should be displayed.

        For calibration, we should display it after the target prompt (index = 1).

        Returns:
            int: The index at which to display the preview (1 for calibration).
        """
        return 1
