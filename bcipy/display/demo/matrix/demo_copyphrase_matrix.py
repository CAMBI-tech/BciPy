"""Demo using the Matrix display with Copy Phrase. This logic is integrated as
a task, but this module demonstates how to customize or as a proof of concept
prior to integration.
"""

from psychopy import core

from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.matrix.display import MatrixDisplay

font = "Overpass Mono"
info = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=[font],
    info_text=['Matrix Copy Phrase Demo'],
)
task_height = 0.1

inter_stim_buffer = .5

stim_properties = StimuliProperties(stim_font=font,
                                    stim_pos=(-0.6, 0.4),
                                    stim_height=0.17,
                                    is_txt_stim=True,
                                    layout='ALP')

# Initialize Stimulus
window_parameters = {
    'full_screen': False,
    'window_height': 500,
    'window_width': 500,
    'stim_screen': 0,
    'background_color': 'black'
}

# Create the stimuli to be presented, in real-time these stimuli will likely be given by a model or randomized

ele_sti = [['+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '<', '_'],
           ['+', 'F', 'G', 'E', '_', 'S', 'Q', 'W', 'E', '<', 'A'],
           ['+', 'F', 'G', 'E', '_', 'S', 'Q', 'W', 'R', '<', 'A'],
           ['+', 'F', 'G', 'E', '_', 'S', 'Q', 'W', 'E', '<', 'A']]
color_sti = [[
    'red', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
    'white', 'white', 'white'
]] * 4

timing_sti = [[2] + [0.25] * 10] * 4

spelled_text = ['COPY_PHA', 'COPY_PH']
task_color = [['white'] * 5 + ['green'] * 2 + ['red'],
              ['white'] * 5 + ['green'] * 2]

# Initialize decision
decisions = ['<', 'R']

# Initialize Window
experiment_clock = core.Clock()
win = init_display_window(window_parameters)

# This is useful during time critical portions of the code, turn off otherwise
win.recordFrameIntervals = False

frameRate = win.getActualFrameRate()

print(frameRate)

task_bar = CopyPhraseTaskBar(win,
                             task_text='COPY_PHRASE',
                             spelled_text='COPY_PHA',
                             colors=['white', 'green'],
                             font=font)
preview_config = PreviewParams(show_preview_inquiry=True,
                               preview_inquiry_length=2,
                               preview_inquiry_key_input='return',
                               preview_inquiry_progress_method=0,
                               preview_inquiry_isi=1)
display = MatrixDisplay(win,
                        experiment_clock,
                        stim_properties,
                        task_bar,
                        info,
                        should_prompt_target=False,
                        preview_config=preview_config)

counter = 0

for spelled in spelled_text:

    display.update_task_bar(text=spelled)
    display.draw_static()
    win.flip()

    for idx in range(int(len(ele_sti) / 2)):
        # Schedule a inquiry
        display.schedule_to(stimuli=ele_sti[counter],
                            timing=timing_sti[counter],
                            colors=color_sti[counter])
        core.wait(inter_stim_buffer)
        inquiry_timing = display.do_inquiry()
        core.wait(inter_stim_buffer)
        counter += 1

    display.draw_static()
    win.flip()
    core.wait(2.0)

win.close()
