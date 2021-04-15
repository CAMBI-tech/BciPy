from psychopy import visual
from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.task import SPACE_CHAR


class CalibrationDisplay(RSVPDisplay):
    """Calibration Display."""

    def __init__(self,
                 window,
                 static_clock,
                 experiment_clock,
                 stimuli,
                 task_display,
                 info,
                 marker_writer=None,
                 trigger_type='image',
                 space_char=SPACE_CHAR,
                 full_screen=False):

        # TODO require user to set these as an array
        info.info_color = [info.info_color]
        info.info_font = [info.info_font]
        info.info_text = [info.info_text]
        info.info_pos = [info.info_pos]
        info.info_height = [info.info_height]

        tmp = visual.TextStim(win=window, font=task_display.task_font, text=task_display.task_text)
        x_task_pos = tmp.boundingBox[0] / window.size[0] - 1
        task_display.task_pos = (x_task_pos, 1 - task_display.task_height)

        super(CalibrationDisplay, self).__init__(
            window,
            static_clock,
            experiment_clock,
            stimuli,
            task_display,
            info,
            marker_writer=marker_writer,
            trigger_type=trigger_type,
            space_char=space_char,
            full_screen=full_screen)
