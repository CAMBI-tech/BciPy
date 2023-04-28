from bcipy.display.paradigm.rsvp.display import RSVPDisplay
from bcipy.helpers.symbols import SPACE_CHAR


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
                 preview_inquiry=None,
                 space_char=SPACE_CHAR,
                 full_screen=False):

        super().__init__(window,
                         static_clock,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         trigger_type=trigger_type,
                         preview_inquiry=preview_inquiry,
                         space_char=space_char,
                         full_screen=full_screen)
