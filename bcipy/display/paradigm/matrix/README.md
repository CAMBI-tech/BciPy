# Matrix Display

The matrix display presents a list of symbols in a grid format. Grid items are evenly spaced. All grid items are always displayed. The matrix can be drawn at different opacities and with different symbols highlighted.

## Basic usage

A matrix display needs a psychopy Window, core.Clock, and configuration for the stimuli (StimuliProperties), task_bar (TaskBar), and information elements (InformationProperties).

```
from psychopy import core

import bcipy.display.components.layout as layout
from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay

win = init_display_window(window_parameters)
task_bar = CalibrationPhraseTaskBar(win, inquiry_count=100)
stim_properties = StimuliProperties(stim_font=font,
                                    stim_pos=[],
                                    stim_height=0.5)
info = InformationProperties(
    info_color=['white'],
    info_pos=[layout.at_bottom(win, height=0.21).center],
    info_height=[0.1],
    info_font=[font],
    info_text=['Matrix Calibration Demo'],
)

matrix_display = MatrixDisplay(win,
                               core.Clock(),
                               stim_properties,
                               task_bar=task_bar,
                               info=info)

matrix_display.draw(grid_opacity=matrix_display.full_grid_opacity,
                    grid_color=matrix_display.grid_color,
                    duration=10)
```

## Specify rows and columns

The number of rows and columns can be specified by using the provided parameters. The number of rows x columns must be greater than or equal to the number of symbols.

When using a task, the `matrix_rows` and `matrix_columns` parameters are used for customization.

```
matrix_display = MatrixDisplay(win,
                               experiment_clock,
                               stim_properties,
                               task_bar=task_bar,
                               info=info,
                               rows=4,
                               columns=7)
```

## Sorting stimuli

A sort order function for the symbols can specified. The sort function includes the ability to provide blank spaces within the grid.

```
from bcipy.helpers.symbols import qwerty_order
matrix_display = MatrixDisplay(win,
                               experiment_clock,
                               stim_properties,
                               task_bar=task_bar,
                               info=info,
                               rows=3,
                               columns=10,
                               sort_order=qwerty_order(is_txt_stim=True))
```

## Layout

Symbol positions are calculated when the display is initialized. The grid will be centered within the window. By default the grid will take up 75% of the width, and 80% of the height, or whichever is smaller depending on the aspect ratio of your monitor. These values can be adjusted by provided a width_pct and height_pct parameter.

When using a task, the `matrix_width` parameter is used for customization.

```
# determine matrix height based on the size of the task_bar
matrix_height_pct = 1 - (2 * task_bar.height_pct)
matrix_display = MatrixDisplay(win,
                               experiment_clock,
                               stim_properties,
                               task_bar=task_bar,
                               info=info,
                               rows=4,
                               columns=7,
                               width_pct=0.7,
                               height_pct=matrix_height_pct)
```

## Symbol positions

Symbol positions are logged after calculation.

## Troubleshooting

You may need to do some trial and error to determine the best matrix configuration for your display. The following demos are helpful for experimenting with different layouts.

* bcipy/display/demo/matrix/demo_matrix_layout.py
* bcipy/display/demo/components/demo_task_bar.py (useful if task_bar is cutoff and the task_padding needs adjustment)
* bcipy/display/demo/components/demo_layouts.py


## Recommended parameters

For tasks which use the matrix display, the following parameters are recommended:

```
    time_fixation: 2
    stim_height: 0.17
    task_height: 0.1
    task_padding: 0.05
```