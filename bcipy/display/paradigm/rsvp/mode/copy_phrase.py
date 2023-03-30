from psychopy import visual
from bcipy.display.paradigm.rsvp.display import RSVPDisplay, BCIPY_LOGO_PATH
from bcipy.helpers.symbols import SPACE_CHAR
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
            task_bar,
            info,
            starting_spelled_text='',
            trigger_type='image',
            space_char=SPACE_CHAR,
            preview_inquiry=None,
            full_screen=False):
        """ Initializes Copy Phrase Task Objects """
        self.starting_spelled_text = starting_spelled_text

        super().__init__(window,
                         clock,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         trigger_type=trigger_type,
                         space_char=space_char,
                         preview_inquiry=preview_inquiry,
                         full_screen=full_screen)

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
