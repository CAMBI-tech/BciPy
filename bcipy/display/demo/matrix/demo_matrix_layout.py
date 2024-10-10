"""Demo Matrix Display functionality related to Calibration task logic."""
# pylint: disable=invalid-name
from psychopy import core

import bcipy.display.components.layout as layout
from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay

# from bcipy.helpers.symbols import qwerty_order

font = 'Overpass Mono Medium'
# Initialize Stimulus
window_parameters = {
    'full_screen': True,
    'window_height': 500,
    'window_width': 700,
    'stim_screen': 1,
    'background_color': 'black'
}

experiment_clock = core.Clock()
win = init_display_window(window_parameters)
win.recordFrameIntervals = False

task_bar = CopyPhraseTaskBar(win,
                             task_text='HELLO_WORLD',
                             spelled_text='HELLO',
                             font=font, colors=['green'],
                             height=0.1, padding=0.15)

matrix_height_pct = 1 - (2 * task_bar.height_pct)

stim_properties = StimuliProperties(stim_font=font,
                                    stim_pos=[],
                                    stim_height=0.5,
                                    is_txt_stim=True)

info = InformationProperties(
    info_color=['white'],
    info_pos=[layout.at_bottom(win, height=0.21).center],
    info_height=[0.1],
    info_font=[font],
    info_text=['Matrix Calibration Demo'],
)

matrix_display = MatrixDisplay(win,
                               experiment_clock,
                               stim_properties,
                               task_bar=task_bar,
                               info=info,
                               rows=4,
                               columns=7,
                               width_pct=0.7,
                               height_pct=matrix_height_pct)
#    sort_order=qwerty_order(is_txt_stim=True))

matrix_display.draw(grid_opacity=matrix_display.full_grid_opacity,
                    grid_color=matrix_display.grid_color,
                    duration=10)
