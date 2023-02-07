from bcipy.display.paradigm.rsvp.display import RSVPDisplay
from bcipy.language.main import SPACE_CHAR


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 stimuli,
                 task_display,
                 info,
                 trigger_type='image',
                 preview_inquiry=None,
                 space_char=SPACE_CHAR,
                 full_screen=False):

        super(CalibrationDisplay, self).__init__(
            window,
            static_clock,
            experiment_clock,
            stimuli,
            task_display,
            info,
            trigger_type=trigger_type,
            preview_inquiry=preview_inquiry,
            space_char=space_char,
            full_screen=full_screen)
