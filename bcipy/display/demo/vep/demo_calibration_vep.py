"""Demo VEP display"""
import logging
import sys
from typing import Any, List

from bcipy.display import (InformationProperties, VEPStimuliProperties,
                           init_display_window)
from bcipy.display.components.layout import centered
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.vep.codes import (DEFAULT_FLICKER_RATES,
                                              ssvep_to_code)
from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.display.paradigm.vep.layout import BoxConfiguration
from bcipy.helpers.clock import Clock
from bcipy.helpers.system_utils import get_screen_info

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

task_text = ['1/3', '2/3', '3/3']
stim_screen = 0
window_parameters = {
    'full_screen': False,
    'window_height': 700,
    'window_width': 700,
    'stim_screen': stim_screen,
    'background_color': 'black'
}
win = init_display_window(window_parameters)
win.recordFrameIntervals = True
frame_rate = win.getActualFrameRate()
if not frame_rate:
    # Allow the demo to work using the configured rate.
    frame_rate = get_screen_info(stim_screen).rate

print(f'Monitor refresh rate: {frame_rate} Hz')

stim_color = [
    'green', 'red', '#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3',
    '#FF8232'
]

# Note: these rates work for a 60hz display
flicker_rates = DEFAULT_FLICKER_RATES
codes = [
    ssvep_to_code(refresh_rate=int(frame_rate), flicker_rate=hz)
    for hz in flicker_rates
]

layout = centered(width_pct=0.95, height_pct=0.80)
box_config = BoxConfiguration(layout, height_pct=0.30)

experiment_clock = Clock()
len_stimuli = 10
stim_props = VEPStimuliProperties(
    stim_font=font,
    stim_pos=box_config.positions,
    stim_height=0.1,
    timing=[4, 0.5, 4],  # target, fixation, stimuli
    stim_color=stim_color,
    inquiry=[],
    stim_length=1,  # how many times to stimuli
    animation_seconds=2.0)
task_bar = CalibrationTaskBar(win, inquiry_count=3, current_index=0, font=font)
vep = VEPDisplay(win,
                 experiment_clock,
                 stim_props,
                 task_bar,
                 info,
                 box_config=box_config,
                 codes=codes,
                 should_prompt_target=True,
                 frame_rate=frame_rate)
wait_seconds = 2

inquiries: List[List[Any]] = [[
    'U', '+', ['C', 'M', 'S'], ['D', 'P', 'X', '_'], ['L', 'U', 'Y'],
    ['E', 'K', 'O'], ['<', 'A', 'F', 'H', 'I', 'J', 'N', 'Q', 'R', 'V', 'Z'],
    ['B', 'G', 'T', 'W']
],
    [
    'D', '+', ['O', 'X'], ['D'], ['P', 'U'],
    ['<', 'B', 'E', 'G', 'H', 'J', 'K', 'L', 'R', 'T'],
    ['A', 'C', 'F', 'I', 'M', 'N', 'Q', 'V', 'Y', '_'],
    ['S', 'W', 'Z']
],
    [
    'S', '+', ['A', 'J', 'K', 'T', 'V', 'W'], ['S'], ['_'],
    ['E', 'G', 'M', 'R'],
    [
        '<', 'B', 'C', 'D', 'H', 'I', 'L', 'N', 'O', 'P', 'Q',
        'U', 'X', 'Z'
    ], ['F', 'Y']
]]

timing = []
# loop over the text and colors, present the stimuli and record the timing
for i, txt in enumerate(task_text):
    vep.update_task_bar(txt)
    inq = inquiries[i]
    vep.schedule_to(stimuli=inq)
    timing += vep.do_inquiry()

print(timing)
win.close()
