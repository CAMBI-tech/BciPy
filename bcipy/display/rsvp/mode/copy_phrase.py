from psychopy import visual
from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.task import SPACE_CHAR

""" RSVP Tasks are RSVPDisplay objects with different structure. They share
    the tasks and the essential elements and stimuli. However layout, length of
    stimuli list, update procedures and colors are different. Therefore each
    mode should be separated from each other carefully.
    Functions:
        update_task_state: update task information of the module """


class CopyPhraseDisplay(RSVPDisplay):
    """ Copy Phrase display object of RSVP
        Attr:
            static_task(visual_Text_Stimuli): aim string of the copy phrase.
                (Stored in self.text[0])
            information(visual_Text_Stimuli): information text. (Stored in
                self.text[1])
            task(Multiinfo_color_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text.
    """

    def __init__(
            self,
            window,
            clock,
            experiment_clock,
            marker_writer,
            static_task_text='COPY_PHRASE',
            static_task_color='White',
            info_text='Press Space Bar to Pause',
            info_color='White',
            info_pos=(0, -.9),
            info_height=0.2,
            info_font='Times',
            task_color=['white'] * 4 + ['green'] * 2 + ['red'],
            task_font='Times',
            task_text='COPY_PH',
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
        """ Initializes Copy Phrase Task Objects """

        tmp = visual.TextStim(win=window, font=task_font,
                              text=static_task_text)
        static_task_pos = (
            tmp.boundingBox[0] / window.size[0] - 1, 1 - task_height)

        info_color = [static_task_color, info_color]
        info_font = [task_font, info_font]
        info_text = [static_task_text, info_text]
        info_pos = [static_task_pos, info_pos]
        info_height = [task_height, info_height]

        # Adjust task position wrt. static task position. Definition of
        # dummy texts are required. Place the task on bottom
        tmp2 = visual.TextStim(win=window, font=task_font, text=task_text)
        x_task_pos = tmp2.boundingBox[0] / window.size[0] - 1
        task_pos = (x_task_pos, static_task_pos[1] - task_height)

        super(CopyPhraseDisplay, self).__init__(
            window, clock,
            experiment_clock,
            marker_writer,
            task_color=task_color,
            task_font=task_font,
            task_pos=task_pos,
            task_height=task_height,
            task_text=task_text,
            info_color=info_color,
            info_font=info_font,
            info_pos=info_pos,
            info_height=info_height,
            info_text=info_text,
            stim_font=stim_font,
            stim_pos=stim_pos,
            stim_height=stim_height,
            stim_sequence=stim_sequence,
            stim_colors=stim_colors,
            stim_timing=stim_timing,
            is_txt_stim=is_txt_stim,
            trigger_type=trigger_type,
            space_char=space_char)

    def update_task_state(self, text, color_list):
        """ Updates task state of Copy Phrase Task by removing letters or
            appending to the right.
            Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each """
        # An empty string will cause an error when we attempt to find its
        # bounding box.
        txt = text if len(text) > 0 else ' '
        tmp2 = visual.TextStim(win=self.window, font=self.task.font, text=txt)
        x_task_pos = tmp2.boundingBox[0] / self.window.size[0] - 1
        task_pos = (x_task_pos, self.text[0].pos[1] - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=task_pos)
