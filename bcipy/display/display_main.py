from psychopy import visual
from bcipy.helpers.system_utils import get_system_info


def init_display_window(parameters):
    """
    Init Display Window.

    Function to Initialize main display window
        needed for all later stimuli presentation.

    See Psychopy official documentation for more information and working demos:
        http://www.psychopy.org/api/visual/window.html
    """

    # Check is full_screen mode is set and get necessary values
    if parameters['full_screen']:

        # get relevant info about the system
        info = get_system_info()

        # set window attributes based on resolution
        window_height = info['RESOLUTION'][1]
        window_width = info['RESOLUTION'][0]

        # set full screen mode to true (removes os dock, explorer etc.)
        full_screen = True

    # otherwise, get user defined window attributes
    else:

        # set window attributes directly from parameters file
        window_height = parameters['window_height']
        window_width = parameters['window_width']

        # make sure full screen is set to false
        full_screen = False

    # Initialize PsychoPy Window for Main Display of Stimuli
    display_window = visual.Window(
        size=[window_width,
              window_height],
        screen=0,
        allowGUI=False,
        useFBO=False,
        fullscr=full_screen,
        allowStencil=False,
        monitor='mainMonitor',
        winType='pyglet', units='norm', waitBlanking=False,
        color=parameters['background_color'])

    # Return display window to caller
    return display_window


class MultiColorText(object):
    """Multi Color Text.

    Implementation of multi color Text Stimuli. Psychopy does not
        support multiple color texts. Draws multiple TextStim elements on
        the screen with different colors.

    Attr:
        texts(list[TextStim]): characters that form the string
    """

    def __init__(self, win, list_color=['red'] * 5, height=0.2,
                 text='dummy_text', font='Times', pos=(0, 0), wrap_width=None,
                 color_space='rgb', opacity=1, depth=-6.0):
        """Initialize Multi Color Text.

        Args:
            win(visual_window): display window
            text(string): string to be displayed
            list_color(list[string]): list of colors of the string
            height(float): height of each character
            pos(tuple): center position of the multi color text

            wrap_width, color_space, opacity, depth : to keep consistency
                 of the visual object definition (required in TextStim)
        """
        self.win = win
        self.pos = pos
        self.text = text
        self.font = font
        self.height = height
        self.list_color = list_color
        self.wrap_width = wrap_width
        self.color_space = color_space
        self.opacity = opacity
        self.depth = depth

        self.texts = []

        # Align characters using pixel wise operations
        width_total_in_pix = 0
        for idx in range(len(list_color)):
            self.texts.append(
                visual.TextStim(win=win, color=list_color[idx], height=height,
                                text=text[idx], font=font, pos=(0, 0),
                                wrapWidth=self.wrap_width,
                                colorSpace=self.color_space,
                                opacity=opacity, depth=depth))
            # Bounding box provides pixel information of each letter
            width_total_in_pix += self.texts[idx].boundingBox[0]

        # Window goes from [-1,1], therefore we need to multiply by 2
        x_pos_text = pos[0] - (width_total_in_pix / win.size[0])
        for idx in range(len(list_color)):
            len_txt = self.texts[idx].boundingBox[0] / win.size[0]
            self.texts[idx].pos = (x_pos_text + len_txt, pos[1])
            x_pos_text += len_txt * 2

    def draw(self):
        """Draw Multi Color Text."""
        for idx in range(len(self.texts)):
            self.texts[idx].draw()

    def update(self, text, color_list, pos):
        """Update (Re-creates) Multicolor Text Object.

        It is more
            compact to erase the previous one and recreate a new object.
            Args:
                text(string): string to be displayed
                color_list(list[string]): list of colors of the string
                pos(tuple): position of the multicolor text
        """
        # Align characters using pixel wise operations
        width_total_in_pix = 0
        self.texts = []
        for idx in range(len(color_list)):
            self.texts.append(
                visual.TextStim(win=self.win, color=color_list[idx],
                                height=self.height,
                                text=text[idx], font=self.font, pos=(0, 0),
                                wrapWidth=self.wrap_width,
                                colorSpace=self.color_space,
                                opacity=self.opacity, depth=self.depth))
            # Bounding box provides pixel information of each letter
            width_total_in_pix += self.texts[idx].boundingBox[0]

        # Window goes from [-1,1], therefore we need to multiply by 2
        x_pos_text = pos[0] - (width_total_in_pix / self.win.size[0])
        for idx in range(len(color_list)):
            len_txt = self.texts[idx].boundingBox[0] / self.win.size[0]
            self.texts[idx].pos = (x_pos_text + len_txt, pos[1])
            x_pos_text += len_txt * 2
