from bcipy.display.paradigm.rsvp.display import RSVPDisplay
from bcipy.core.symbols import SPACE_CHAR


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 stimuli,
                 task_bar,
                 info,
                 trigger_type='image',
                 preview_config=None,
                 space_char=SPACE_CHAR,
                 full_screen=False):

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
        """
        return 1
