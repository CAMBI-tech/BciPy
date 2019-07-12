from psychopy import visual
from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.task import SPACE_CHAR


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 marker_writer,
                 info_text='Press Space Bar to Pause',
                 info_color='White',
                 info_pos=(0, -.9),
                 info_height=0.2,
                 info_font='Times',
                 task_color=['white'],
                 task_font='Times',
                 task_text='1/100',
                 task_height=0.1,
                 stim_font='Times',
                 stim_pos=(-.8, .9),
                 stim_height=0.2,
                 stim_sequence=['a'] * 10,
                 stim_colors=['white'] * 10,
                 stim_timing=[1] * 10,
                 is_txt_stim=True,
                 trigger_type='image',
                 space_char=SPACE_CHAR):

        info_color = [info_color]
        info_font = [info_font]
        info_text = [info_text]
        info_pos = [info_pos]
        info_height = [info_height]

        tmp = visual.TextStim(win=window, font=task_font, text=task_text)
        x_task_pos = tmp.boundingBox[0] / window.size[0] - 1
        task_pos = (x_task_pos, 1 - task_height)

        super(CalibrationDisplay, self).__init__(
            window,
            static_clock,
            experiment_clock,
            marker_writer,
            task_color=task_color,
            task_font=task_font,
            task_pos=task_pos,
            task_height=task_height,
            task_text=task_text,
            info_color=info_color,
            info_text=info_text,
            info_font=info_font,
            info_pos=info_pos,
            stim_font=stim_font,
            stim_pos=stim_pos,
            stim_height=stim_height,
            stim_sequence=stim_sequence,
            stim_colors=stim_colors,
            stim_timing=stim_timing,
            is_txt_stim=is_txt_stim,
            trigger_type=trigger_type,
            space_char=space_char)
