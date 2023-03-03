from bcipy.display.paradigm.rsvp.display import RSVPDisplay
from bcipy.helpers.task import SPACE_CHAR
from bcipy.display.components.task_bar import CalibrationTaskBar, TaskBar


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 stimuli,
                 task_bar_config,
                 info,
                 trigger_type='image',
                 preview_inquiry=None,
                 space_char=SPACE_CHAR,
                 full_screen=False):

        super().__init__(window,
                         static_clock,
                         experiment_clock,
                         stimuli,
                         task_bar_config,
                         info,
                         trigger_type=trigger_type,
                         preview_inquiry=preview_inquiry,
                         space_char=space_char,
                         full_screen=full_screen)

    def build_task_bar(self) -> TaskBar:
        """Override to make a CalibrationTaskBar"""
        # Initialize to 0 since `update` is called before initial display.
        return CalibrationTaskBar(self.window,
                                  self.task_bar_config,
                                  current_index=0)
