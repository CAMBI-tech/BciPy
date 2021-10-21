from psychopy import visual
from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.task import SPACE_CHAR

"""Note:

RSVP Tasks are RSVPDisplay objects with different structure. They share
the tasks and the essential elements and stimuli. However, layout, length of
stimuli list, update procedures and colors are different. Therefore each
mode should be separated from each other carefully.
Functions:
    update_task_state: update task information of the module
"""


class CopyPhraseDisplay(RSVPDisplay):
    """ Copy Phrase display object of RSVP

        Custom attributes:
            static_task_text(str): target text for the user to attempt to spell
            static_task_color(str): target text color for the user to attempt to spell
    """

    def __init__(
            self,
            window,
            clock,
            experiment_clock,
            stimuli,
            task_display,
            info,
            static_task_text='COPY_PHRASE',
            static_task_color='White',
            trigger_type='image',
            space_char=SPACE_CHAR,
            preview_inquiry=None,
            full_screen=False):
        """ Initializes Copy Phrase Task Objects """

        tmp = visual.TextStim(win=window, font=task_display.task_font,
                              text=static_task_text)
        static_task_pos = (
            tmp.boundingBox[0] / window.size[0] - 1, 1 - task_display.task_height)

        info.info_color = [static_task_color, info.info_color]
        info.info_font = [task_display.task_font, info.info_font]
        info.info_text = [static_task_text, info.info_text]
        info.info_pos = [static_task_pos, info.info_pos]
        info.info_height = [task_display.task_height, info.info_height]

        # Adjust task position wrt. static task position. Definition of
        # dummy texts are required. Place the task on bottom
        tmp2 = visual.TextStim(win=window, font=task_display.task_font, text=task_display.task_text)
        x_task_pos = tmp2.boundingBox[0] / window.size[0] - 1
        task_display.task_pos = (x_task_pos, static_task_pos[1] - task_display.task_height)

        super(CopyPhraseDisplay, self).__init__(
            window, clock,
            experiment_clock,
            stimuli,
            task_display,
            info,
            trigger_type=trigger_type,
            space_char=space_char,
            preview_inquiry=preview_inquiry,
            full_screen=full_screen)

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
