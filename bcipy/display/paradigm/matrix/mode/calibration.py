from bcipy.display.paradigm.matrix.display import MatrixDisplay


class CalibrationDisplay(MatrixDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 experiment_clock,
                 stimuli,
                 task_display,
                 info,
                 trigger_type='text',
                 symbol_set=None):

        super(CalibrationDisplay, self).__init__(
            window,
            experiment_clock=experiment_clock,
            stimuli=stimuli,
            task_display=task_display,
            info=info,
            trigger_type=trigger_type,
            symbol_set=symbol_set)
