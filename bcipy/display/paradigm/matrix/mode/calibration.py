from bcipy.display.matrix.display import MatrixDisplay
from bcipy.helpers.task import SPACE_CHAR


class CalibrationDisplay(MatrixDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 stimuli,
                 task_display,
                 info,
                 trigger_type='text',
                 space_char=SPACE_CHAR,
                 full_screen=False,
                 symbol_set=None):

        super(CalibrationDisplay, self).__init__(
            window,
            static_clock,
            experiment_clock,
            stimuli,
            task_display,
            info,
            trigger_type=trigger_type,
            space_char=space_char,
            full_screen=full_screen,
            symbol_set=symbol_set)
