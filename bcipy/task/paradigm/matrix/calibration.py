"""Matrix calibration task module.

This module provides the Matrix calibration task implementation which performs
Matrix stimulus inquiries to elicit ERPs. The task presents a matrix of stimuli
and highlights them according to configured parameters.
"""

from typing import Any, Dict, List, Optional

from psychopy import visual

from bcipy.core.parameters import Parameters
from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.utils import get_screen_info
from bcipy.io.save import save_stimuli_position_info
from bcipy.task.calibration import BaseCalibrationTask


class MatrixCalibrationTask(BaseCalibrationTask):
    """Matrix Calibration Task.

    This task performs Matrix stimulus inquiries to elicit ERPs by highlighting
    elements in a matrix display. Parameters control the number of stimuli,
    highlight duration, colors, and text/image inputs.

    Task flow:
        1. Setup variables
        2. Initialize EEG
        3. Await user input
        4. Setup stimuli
        5. Perform highlighting inquiries
        6. Save data

    Attributes:
        name: Name of the task.
        paradigm: Name of the paradigm.
        parameters: Task configuration parameters.
        file_save: Path for saving task data.
        fake: Whether to run in fake (testing) mode.
        window: PsychoPy window for display.
        experiment_clock: Task timing clock.
        display: Matrix display instance.
        symbol_set: Set of symbols to display.
    """

    name = 'Matrix Calibration'
    paradigm = 'Matrix'

    @property
    def screen_info(self) -> Dict[str, Any]:
        """Get screen properties.

        Returns:
            Dict[str, Any]: Dictionary containing screen size, refresh rate,
                and units information.
        """
        return {
            'screen_size_pixels': self.window.size.tolist(),
            'screen_hz': get_screen_info().rate,
            'screen_units': 'norm',
        }

    def init_display(self) -> MatrixDisplay:
        """Initialize the matrix display.

        Returns:
            MatrixDisplay: Configured matrix display instance.
        """
        return init_matrix_display(self.parameters, self.window,
                                   self.experiment_clock, self.symbol_set)

    def exit_display(self) -> None:
        """Clean up display resources and save screenshot.

        Raises:
            AssertionError: If display is not a MatrixDisplay instance.
        """
        assert isinstance(self.display, MatrixDisplay)
        self.display.capture_grid_screenshot(self.file_save)
        return super().exit_display()

    def cleanup(self) -> None:
        """Perform cleanup operations and save stimuli position data.

        Raises:
            AssertionError: If display is not a MatrixDisplay instance.
        """
        assert isinstance(self.display, MatrixDisplay)
        save_stimuli_position_info(self.display.stim_positions, self.file_save,
                                   self.screen_info)
        return super().cleanup()

    def session_task_data(self) -> Optional[Dict[str, Any]]:
        """Get session task data.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing stimuli positions
                and screen information.

        Raises:
            AssertionError: If display is not a MatrixDisplay instance.
        """
        assert isinstance(self.display, MatrixDisplay)
        return {**self.display.stim_positions, **self.screen_info}


def init_matrix_display(parameters: Parameters, window: visual.Window,
                        experiment_clock: Clock,
                        symbol_set: List[str]) -> MatrixDisplay:
    """Initialize the matrix display with given parameters.

    Args:
        parameters: Task configuration parameters.
        window: PsychoPy window for display.
        experiment_clock: Task timing clock.
        symbol_set: Set of symbols to display.

    Returns:
        MatrixDisplay: Configured matrix display instance.
    """
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(parameters['matrix_stim_pos_x'], parameters['matrix_stim_pos_y']),
                                stim_height=parameters['matrix_stim_height'],
                                stim_inquiry=[''] * parameters['stim_length'],
                                stim_colors=[parameters['stim_color']] *
                                parameters['stim_length'],
                                stim_timing=[10] * parameters['stim_length'],
                                is_txt_stim=parameters['is_txt_stim'],
                                prompt_time=parameters['time_prompt'],
                                layout=parameters['matrix_keyboard_layout'])

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['matrix_task_height'],
                                  padding=parameters['matrix_task_padding'])

    return MatrixDisplay(window,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         rows=parameters['matrix_rows'],
                         columns=parameters['matrix_columns'],
                         width_pct=parameters['matrix_width'],
                         height_pct=1 - (2 * task_bar.height_pct),
                         trigger_type=parameters['trigger_type'],
                         symbol_set=symbol_set,
                         preview_config=parameters.instantiate(PreviewParams))
