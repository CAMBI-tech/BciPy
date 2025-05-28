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

    Calibration task performs an Matrix stimulus inquiry
        to elicit an ERP. Parameters change the number of stimuli
        (i.e. the subset of matrix) and for how long they will highlight.
        Parameters also change color and text / image inputs.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> highlighting inquiries -->
        saving data

    PARAMETERS:
    ----------
    parameters (dict)
    file_save (str)
    fake (bool)

    """
    name = 'Matrix Calibration'
    paradigm = 'Matrix'

    @property
    def screen_info(self) -> Dict[str, Any]:
        """Screen properties"""
        return {
            'screen_size_pixels': self.window.size.tolist(),
            'screen_hz': get_screen_info().rate,
            'screen_units': 'norm',
        }

    def init_display(self) -> MatrixDisplay:
        """Initialize the display"""
        return init_matrix_display(self.parameters, self.window,
                                   self.experiment_clock, self.symbol_set)

    def exit_display(self) -> None:
        assert isinstance(self.display, MatrixDisplay)
        self.display.capture_grid_screenshot(self.file_save)
        return super().exit_display()

    def cleanup(self) -> None:
        assert isinstance(self.display, MatrixDisplay)
        save_stimuli_position_info(self.display.stim_positions, self.file_save,
                                   self.screen_info)
        return super().cleanup()

    def session_task_data(self) -> Optional[Dict[str, Any]]:
        assert isinstance(self.display, MatrixDisplay)
        return {**self.display.stim_positions, **self.screen_info}


def init_matrix_display(parameters: Parameters, window: visual.Window,
                        experiment_clock: Clock,
                        symbol_set: List[str]) -> MatrixDisplay:
    """Initialize the matrix display"""
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
