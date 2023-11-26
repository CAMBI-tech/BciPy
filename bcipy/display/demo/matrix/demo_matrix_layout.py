"""Demo Matrix Display functionality related to Calibration task logic."""
# pylint: disable=invalid-name
from psychopy import core

import bcipy.display.components.layout as layout
from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.helpers.symbols import qwerty_order

# Initialize Stimulus
window_parameters = {
    'full_screen': False,
    'window_height': 500,
    'window_width': 700,
    'stim_screen': 0,
    'background_color': 'black'
}

experiment_clock = core.Clock()
win = init_display_window(window_parameters)
win.recordFrameIntervals = False

task_bar = CalibrationTaskBar(win,
                              inquiry_count=5,
                              current_index=0,
                              font='Arial')

stim_properties = StimuliProperties(stim_font='Arial',
                                    stim_pos=[],
                                    stim_height=0.17,
                                    is_txt_stim=True)

info = InformationProperties(
    info_color=['white'],
    info_pos=[layout.at_bottom(win, height=0.21).center],
    info_height=[0.1],
    info_font=['Arial'],
    info_text=['Matrix Calibration Demo'],
)

matrix_display = MatrixDisplay(win,
                               experiment_clock,
                               stim_properties,
                               task_bar=task_bar,
                               info=info,
                               rows=3,
                               columns=10,
                               width_pct=0.9,
                               sort_order=qwerty_order(is_txt_stim=True))

matrix_display.draw(grid_opacity=matrix_display.full_grid_opacity,
                    grid_color=matrix_display.grid_color,
                    duration=10)
