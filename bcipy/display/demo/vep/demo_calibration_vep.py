import logging
import sys

from psychopy import core, visual

from bcipy.display import (InformationProperties, VEPStimuliProperties,
                           init_display_window)
from bcipy.display.components.layout import centered
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.display.paradigm.vep.layout import BoxConfiguration
from bcipy.helpers.clock import Clock

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
root.addHandler(handler)

font = 'Courier New'
info = InformationProperties(
    info_color=['White'],
    info_pos=[(-.5, -.75)],
    info_height=[0.1],
    info_font=[font],
    info_text=['VEP Display Demo'],
)

task_text = ['1/4', '2/4', '3/4', '4/4']
num_boxes = 6

window_parameters = {
    'full_screen': False,
    'window_height': 700,
    'window_width': 700,
    'stim_screen': 1,
    'background_color': 'black'
}
win = init_display_window(window_parameters)
win.recordFrameIntervals = True
frameRate = win.getActualFrameRate()

print(f'Monitor refresh rate: {frameRate} Hz')

box_colors = ['#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232']
stim_color = [[color] for i, color in enumerate(box_colors) if i < num_boxes]

layout = centered(width_pct=0.95, height_pct=0.80)
box_config = BoxConfiguration(layout, num_boxes=num_boxes)

experiment_clock = Clock()
len_stimuli = 10
stimuli = VEPStimuliProperties(
    stim_color=stim_color,
    stim_pos=box_config.positions,
    stim_height=0.1,
    stim_font=font,
    timing=(1, 0.5, 2, 4),  # prompt, fixation, animation, stimuli
    stim_length=1,  # how many times to stimuli
)
task_bar = CalibrationTaskBar(win,
                              inquiry_count=4,
                              current_index=0,
                              font=font)
vep = VEPDisplay(win, experiment_clock, stimuli, task_bar, info, box_config=box_config)
timing = []
wait_seconds = 2

# loop over the text and colors, present the stimuli and record the timing
for txt in task_text:
    vep.update_task_bar(txt)
    if num_boxes == 4:
        stim = [['A', 'B'], ['Z'], ['P'], ['R', 'W']]
    if num_boxes == 6:
        stim = [['A'], ['B'], ['Z', 'X'], ['P', 'I'], ['R'], ['W', 'C']]
    vep.schedule_to(stimuli=stim)
    timing += vep.do_inquiry()

    # show the wait screen, this will only happen once
    while wait_seconds > 0:
        wait_seconds -= 1
        vep.wait_screen(f"Waiting for {wait_seconds}s", color='white')
        core.wait(1)

print(timing)
win.close()
