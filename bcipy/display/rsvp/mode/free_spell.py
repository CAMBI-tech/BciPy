from psychopy import visual
from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.task import SPACE_CHAR


class FreeSpellingDisplay(RSVPDisplay):
    """ Free Spelling Task object of RSVP
        Attr:
            information(visual_Text_Stimuli): information text.
            task(visual_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock, experiment_clock, marker_writer,
                 info_text='Press Space Bar to Pause',
                 info_color='White', info_pos=(0, -.9),
                 info_height=0.2, info_font='Times',
                 task_color=['white'],
                 task_font='Times', task_text='1/100', task_height=0.1,
                 stim_font='Times', stim_pos=(-.8, .9), stim_height=0.2,
                 stim_sequence=['a'] * 10, stim_colors=['white'] * 10,
                 stim_timing=[1] * 10,
                 is_txt_stim=True,
                 trigger_type='image',
                 space_char=SPACE_CHAR):
        """ Initializes Free Spelling Task Objects """

        info_color = [info_color]
        info_font = [info_font]
        info_text = [info_text]
        info_pos = [info_pos]
        info_text = [info_height]

        tmp = visual.TextStim(win=window, font=task_font, text=task_text)
        x_task_pos = tmp.boundingBox[0] / window.size[0] - 1
        task_pos = (x_task_pos, 1 - task_height)

        super(FreeSpellingDisplay, self).__init__(
            window, clock,
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