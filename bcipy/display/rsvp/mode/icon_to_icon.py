from psychopy import visual
from bcipy.display.rsvp.display import RSVPDisplay
from bcipy.helpers.stimuli import resize_image
from bcipy.helpers.task import SPACE_CHAR


class IconToIconDisplay(RSVPDisplay):
    """ Icon to Icon or Icon to Word task display
        is_word determines whether this is an icon to word matching task"""

    def __init__(self, window, clock,
                 experiment_clock,
                 marker_writer,
                 info_text='Press Space Bar to Pause',
                 info_color='White',
                 info_pos=(0, -.9),
                 info_height=0.2,
                 info_font='Times',
                 task_color=['white'],
                 task_font='Times',
                 task_height=0.1,
                 stim_font='Times',
                 stim_pos=(-.8, .9),
                 stim_height=0.2,
                 stim_sequence=['a'] * 10,
                 stim_colors=['white'] * 10,
                 stim_timing=[1] * 10,
                 is_txt_stim=True,
                 trigger_type='image',
                 is_word=False):
        """ Initializes Icon Matching Task Objects """

        info_color = [info_color]
        info_font = [info_font]
        info_text = [info_text]
        info_pos = [info_pos]
        info_text = [info_height]

        if is_word:
            task_height *= 2

        tmp = visual.TextStim(win=window, font=task_font, text=' ')
        x_task_pos = tmp.boundingBox[0] / window.size[0] - 1

        self.task_pos = (x_task_pos, 1 - task_height)
        self.stim_height = stim_height
        self.stim_pos = stim_pos

        super(IconToIconDisplay, self).__init__(
            window, clock,
            experiment_clock,
            marker_writer,
            task_color=task_color,
            task_font=task_font,
            task_pos=self.task_pos,
            task_height=task_height,
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
            trigger_type=trigger_type)

        self.is_word = is_word

        if not is_word:
            self.rect = visual.Rect(
                win=window,
                width=task_height,
                height=task_height,
                lineColor='black',
                pos=(stim_pos),
                lineWidth=10,
                ori=0.0)
            self.rect_drawn_frames = 0

            self.task = visual.ImageStim(
                win=window, image=None, mask=None,
                units='', pos=self.task_pos,
                size=(task_height * 2, task_height * 2),
                ori=0.0)

            self.target_text = visual.TextStim(
                win=window,
                color='yellow',
                text='TARGET:',
                pos=(stim_pos[0] - 0.5, stim_pos[1]),
                height=task_height)

    def draw_static(self):
        if not self.is_word:
            """Draw static elements in a stimulus."""
            if(self.rect_drawn_frames < self.time_to_present):
                self.rect.draw()
                self.target_text.draw()
                self.rect_drawn_frames += 1

        super(IconToIconDisplay, self).draw_static()

    def update_task_state(
            self,
            image_path,
            task_height,
            rect_color,
            window_size,
            is_word):
        """ Updates task state of Icon to Icon/Word Matching Task by changing the
        image or text displayed at the top of the screen.
        Also updates rectangle size.
            Args:
                image_path(str): the path to the image to be displayed
                task_height(int): the height of the task image
                rect_color(str): the color of the rectangle
                window_size(tuple): The size of the window
                is_word(bool): word matching task
        """

        if is_word:
            # Display text at top of screen if we are matching icons to words
            txt = image_path if len(image_path) > 0 else ' '
            tmp2 = visual.TextStim(win=self.window, font=self.task.font, text=txt)
            x_task_pos = (tmp2.boundingBox[0] * 2.2) / self.window.size[0] - 1
            self.task_pos = (x_task_pos, self.task_pos[1])
            self.update_task(text=txt, color_list=['white'], pos=self.task_pos)
        else:
            # Otherwise, display an image at the top of the screen
            self.task.image = image_path

            image_width, image_height = resize_image(
                image_path,
                window_size,
                task_height)

            self.target_text.pos = (
                self.stim_pos[0] - image_width - 0.5,
                self.stim_pos[1])

            self.task.pos = (
                self.task_pos[0] + image_width * 2,
                self.task_pos[1] - image_width / 2)
            self.task.size = (image_width * 2, image_height * 2)

            self.rect_drawn_frames = 0
            self.rect.width = image_width / task_height * self.stim_height
            self.rect.height = image_height / task_height * self.stim_height
            self.rect.lineColor = rect_color
