from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.display.components.task_bar import CalibrationTaskBar, TaskBar


class MatrixCalibrationDisplay(MatrixDisplay):
    """Calibration Display."""

    def build_task_bar(self) -> TaskBar:
        """Creates a TaskBar"""
        return CalibrationTaskBar(self.window,
                                  self.task_bar_config,
                                  current_index=0)
