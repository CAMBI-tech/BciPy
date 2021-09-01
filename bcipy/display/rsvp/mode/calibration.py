from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.task import SPACE_CHAR


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 stimuli,
                 task_display,
                 info,
                 marker_writer=None,
                 trigger_type='image',
                 space_char=SPACE_CHAR,
                 full_screen=False):

        super(CalibrationDisplay, self).__init__(
            window,
            static_clock,
            experiment_clock,
            stimuli,
            task_display,
            info,
            marker_writer=marker_writer,
            trigger_type=trigger_type,
            space_char=space_char,
            full_screen=full_screen)
