# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import visual
from utils.get_system_info import get_system_info


def init_display_window(parameters):
    """
    Init Display Window.

    Function to Initialize main display window
        needed for all later stimuli presentation.

    See Psychopy official documentation for more information and working demos:
        http://www.psychopy.org/api/visual/window.html
    """

    # Check is full_screen mode is set and get necessary values
    if parameters['full_screen']['value'] == 'true':

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
        window_height = parameters['window_height']['value']
        window_width = parameters['window_width']['value']

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
        winType='pyglet', units='norm', waitBlanking=True,
        color=parameters['background_color']['value'])

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


class BarGraph(object):
    """Bar Graph object for RSVP Display.

    Attr:
        texts(list[visual_Text_Stimuli]): items to show
        bars(list[visual_Rect_Stimuli]): corresponding density bars
    """

    def __init__(self, win, tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5),
                 size_domain=10, color_txt='white', font_bg='Times',
                 color_bar_bg='white', max_num_step=20):
        """Initialize Bar Graph.

        Args:
            win(visual_window): display window
            tr_pos_bg(tuple) - bl_pos_bg(tuple): bar graph lies in a
            rectangular region in window tr(top right) and bl(bottom
            left) are tuples of (x,y) coordinates of corresponding edges
            size_domain(int): number of items to be shown
            color_txt(string): color of the letters
            font_bg(string): font of letters
            color_bar_bg(string): color of density bars
            max_num_step(int): maximum number of steps for animation
        """

        self.win = win
        self.bl_pos = bl_pos_bg
        self.tr_pos = tr_pos_bg
        self.size_domain = size_domain

        letters = ['a'] * size_domain

        self.height_text_bg = (tr_pos_bg[1] - bl_pos_bg[1]) / self.size_domain
        # TODO: insert aspect ratio parameter
        self.width_text_bg = 0.8 * abs(self.height_text_bg)

        self.texts, self.bars = [], []
        for idx in range(size_domain):
            shift = idx * self.height_text_bg
            pos_text = tuple([self.bl_pos[0] + self.width_text_bg / 2,
                              self.bl_pos[
                                  1] + self.height_text_bg / 2 + shift])
            pos_bar = tuple(
                [(self.tr_pos[0] + self.bl_pos[0] + self.width_text_bg) / 2,
                 self.bl_pos[1] + self.height_text_bg / 2 + shift])
            width_bar = (pos_bar[0] - (
                self.bl_pos[0] + self.width_text_bg)) * 2
            self.texts.append(
                visual.TextStim(win=win, color=color_txt,
                                height=self.height_text_bg, text=letters[idx],
                                font=font_bg, pos=pos_text, wrapWidth=None,
                                colorSpace='rgb', opacity=1, depth=-6.0))
            self.bars.append(
                visual.Rect(win=win, width=width_bar,
                            height=self.height_text_bg,
                            fillColor=color_bar_bg, fillColorSpace='rgb',
                            lineColor=None,
                            pos=pos_bar))

        self.weight_bars = [0] * self.size_domain
        self.scheduled_arg = self.weight_bars
        self.scheduled_weight = letters
        self.max_num_step = max_num_step

    def update(self, letters, weight):
        """Update bar graph parameters.

        Args:
            letters(list[char]): characters to be displayed
            weight(list[float]): densities of characters to be displayed
        """
        for idx in range(self.size_domain):
            shift = idx * self.height_text_bg
            x_bar = (self.bl_pos[0] + self.width_text_bg) + (weight[idx] * (
                self.tr_pos[0] - (self.bl_pos[0] + self.width_text_bg))) / 2
            pos_bar = tuple([x_bar,
                             self.bl_pos[1] + self.height_text_bg / 2 + shift])
            width_bar = (pos_bar[0] - (
                self.bl_pos[0] + self.width_text_bg)) * 2
            self.texts[idx].text = letters[idx]
            self.bars[idx].pos = pos_bar
            self.bars[idx].width = width_bar

    def draw(self):
        """Draw Bar Graph."""
        for idx in range(self.size_domain):
            self.texts[idx].draw()
            self.bars[idx].draw()

    def schedule_to(self, letters, weight):
        """Schedule Bar Graph.

        Args:
            letters(list[char]): characters to be displayed
            weight(list[float]): densities of characters to be displayed
        """
        self.scheduled_arg = letters
        self.scheduled_weight = list(
            np.array(weight) / np.sum(np.array(weight)))

    def animate(self, step):
        """Animate Bar Graph.

        Args:
            step(int): <max_num_step, >0, updates to given step number
        """

        weight_ani = []
        for idx in range(self.size_domain):
            weight_ani = list(np.asarray(self.weight_bars) + (
                np.asarray(self.scheduled_weight) - np.asarray(
                    self.weight_bars)) / self.max_num_step * step)
            self.update(self.scheduled_arg, weight_ani)
            self.draw()

        self.weight_bars = weight_ani

    def reset_weights(self):
        """Reset Bar Graph Weights."""
        self.weight_bars = list(np.ones(self.size_domain) / self.size_domain)
