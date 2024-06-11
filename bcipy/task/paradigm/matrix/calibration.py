from typing import Any, Dict, Optional

from psychopy import core, visual

from bcipy.display import Display, InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import save_stimuli_position_info
from bcipy.helpers.system_utils import get_screen_info
from bcipy.task.base_calibration import BaseCalibrationTask


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
    win (PsychoPy Display Object)
    daq (Data Acquisition Object [ClientManager]])
    parameters (Parameters Object)
    file_save (String)
    """
    MODE = 'Matrix'

    @property
    def screen_info(self) -> Dict[str, Any]:
        """Screen properties"""
        return {
            'screen_size_pixels': self.window.size.tolist(),
            'screen_hz': get_screen_info().rate,
            'screen_units': 'norm',
        }

    def init_display(self) -> Display:
        """Initialize the display"""
        return init_matrix_display(self.parameters, self.window,
                                   self.experiment_clock, self.symbol_set)

    def exit_display(self) -> None:
        self.display.capture_grid_screenshot(self.file_save)
        return super().exit_display()

    def cleanup(self) -> None:
        # TODO: refactor offline_analysis to use session data and and remove this.
        save_stimuli_position_info(self.display.stim_positions, self.file_save,
                                   self.screen_info)
        return super().cleanup()

    def session_task_data(self) -> Optional[Dict[str, Any]]:
        return {**self.display.stim_positions, **self.screen_info}


def init_matrix_display(parameters: Parameters, window: visual.Window,
                        experiment_clock: Clock,
                        symbol_set: core.StaticPeriod) -> MatrixDisplay:
    """Initialize the matrix display"""
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(-0.6, 0.4),
                                stim_height=0.1,
                                stim_inquiry=[''] * parameters['stim_length'],
                                stim_colors=[parameters['stim_color']] *
                                parameters['stim_length'],
                                stim_timing=[10] * parameters['stim_length'],
                                is_txt_stim=parameters['is_txt_stim'],
                                prompt_time=parameters["time_prompt"])

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height'],
                                  padding=parameters['task_padding'])

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
                         symbol_set=symbol_set)
