from psychopy import visual
from bcipy.display.paradigm.rsvp.display import RSVPDisplay, BCIPY_LOGO_PATH
from bcipy.language.main import SPACE_CHAR
from bcipy.helpers.stimuli import resize_image

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
        self.target_text = static_task_text
        static_task_pos = (0, 1 - task_display.task_height)

        info.info_color.append(static_task_color)
        info.info_font.append(task_display.task_font)
        info.info_text.append(static_task_text)
        info.info_pos.append(static_task_pos)
        info.info_height.append(task_display.task_height)

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
                color_list(list[string]): list of colors for each
        """
        # Add padding to attempt aligning the task and information text. *Note:* this only works with monospaced fonts
        padding = abs(len(self.target_text) - len(text))
        text += ' ' * padding
        task_pos = (0, self.info.info_pos[-1][1] - self.task.height)
        self.update_task(text=text, color_list=color_list, pos=task_pos)

    def wait_screen(self, message: str, message_color: str) -> None:
        """Wait Screen.

        Args:
            message(string): message to be displayed while waiting
            message_color(string): color of the message to be displayed
        """

        self.draw_static()

        # Construct the wait message
        wait_message = visual.TextStim(win=self.window,
                                       font=self.stimuli_font,
                                       text=message,
                                       height=.1,
                                       color=message_color,
                                       pos=(0, -.55),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1,
                                       depth=-6.0)

        # try adding the BciPy logo to the wait screen
        try:
            wait_logo = visual.ImageStim(
                self.window,
                image=BCIPY_LOGO_PATH,
                pos=(0, 0),
                mask=None,
                ori=0.0)
            wait_logo.size = resize_image(
                BCIPY_LOGO_PATH,
                self.window.size,
                1)
            wait_logo.draw()

        except Exception as e:
            self.logger.exception(f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            raise e

        # Draw and flip the screen.
        wait_message.draw()
        self.window.flip()
