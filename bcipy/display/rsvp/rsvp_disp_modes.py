
# -*- coding: utf-8 -*-

from psychopy import visual
from PIL import Image
from bcipy.display.rsvp.rsvp_disp import RSVPDisplay

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
            task(Multicolor_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock, experiment_clock, marker_writer,
                 static_text_task='COPY_PHRASE',
                 static_color_task='White',
                 text_info='Press Space Bar to Pause',
                 color_info='White', pos_info=(0, -.9),
                 height_info=0.2, font_info='Times',
                 color_task=['white'] * 4 + ['green'] * 2 + ['red'],
                 font_task='Times', text_task='COPY_PH', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=True,
                 trigger_type='image'):
        """ Initializes Copy Phrase Task Objects """

        tmp = visual.TextStim(win=window, font=font_task,
                              text=static_text_task)
        static_pos_task = (
            tmp.boundingBox[0] / window.size[0] - 1, 1 - height_task)

        color_text = [static_color_task, color_info]
        font_text = [font_task, font_info]
        text_text = [static_text_task, text_info]
        pos_text = [static_pos_task, pos_info]
        height_text = [height_task, height_info]

        # Adjust task position wrt. static task position. Definition of
        # dummy texts are required. Place the task on bottom
        tmp2 = visual.TextStim(win=window, font=font_task, text=text_task)
        x_pos_task = tmp2.boundingBox[0] / window.size[0] - 1
        pos_task = (x_pos_task, static_pos_task[1] - height_task)

        super(CopyPhraseDisplay, self).__init__(
            window, clock,
            experiment_clock,
            marker_writer,
            color_task=color_task,
            font_task=font_task,
            pos_task=pos_task,
            task_height=height_task,
            text_task=text_task,
            color_text=color_text,
            text_text=text_text,
            font_text=font_text,
            pos_text=pos_text,
            height_text=height_text,
            font_sti=font_sti,
            pos_sti=pos_sti,
            sti_height=sti_height,
            stim_sequence=stim_sequence,
            color_list_sti=color_list_sti,
            time_list_sti=time_list_sti,
            tr_pos_bg=tr_pos_bg,
            bl_pos_bg=bl_pos_bg,
            size_domain_bg=size_domain_bg,
            color_bg_txt=color_bg_txt,
            font_bg_txt=font_bg_txt,
            color_bar_bg=color_bar_bg,
            is_txt_sti=is_txt_sti,
            trigger_type=trigger_type)

    def update_task_state(self, text, color_list):
        """ Updates task state of Copy Phrase Task by removing letters or
            appending to the right.
            Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each """
        # An empty string will cause an error when we attempt to find its
        # bounding box.
        txt = text if len(text) > 0 else ' '
        tmp2 = visual.TextStim(win=self.win, font=self.task.font, text=txt)
        x_pos_task = tmp2.boundingBox[0] / self.win.size[0] - 1
        pos_task = (x_pos_task, self.text[0].pos[1] - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=pos_task)


class FreeSpellingDisplay(RSVPDisplay):
    """ Free Spelling Task object of RSVP
        Attr:
            information(visual_Text_Stimuli): information text.
            task(visual_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock, experiment_clock, marker_writer,
                 text_info='Press Space Bar to Pause',
                 color_info='White', pos_info=(0, -.9),
                 height_info=0.2, font_info='Times',
                 color_task=['white'],
                 font_task='Times', text_task='1/100', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=True,
                 trigger_type='image'):
        """ Initializes Free Spelling Task Objects """

        color_text = [color_info]
        font_text = [font_info]
        text_text = [text_info]
        pos_text = [pos_info]
        height_text = [height_info]

        tmp = visual.TextStim(win=window, font=font_task, text=text_task)
        x_pos_task = tmp.boundingBox[0] / window.size[0] - 1
        pos_task = (x_pos_task, 1 - height_task)

        super(FreeSpellingDisplay, self).__init__(
            window, clock,
            experiment_clock,
            marker_writer,
            color_task=color_task,
            font_task=font_task,
            pos_task=pos_task,
            task_height=height_task,
            text_task=text_task,
            color_text=color_text,
            text_text=text_text,
            font_text=font_text,
            pos_text=pos_text,
            height_text=height_text,
            font_sti=font_sti,
            pos_sti=pos_sti,
            sti_height=sti_height,
            stim_sequence=stim_sequence,
            color_list_sti=color_list_sti,
            time_list_sti=time_list_sti,
            tr_pos_bg=tr_pos_bg,
            bl_pos_bg=bl_pos_bg,
            size_domain_bg=size_domain_bg,
            color_bg_txt=color_bg_txt,
            font_bg_txt=font_bg_txt,
            color_bar_bg=color_bar_bg,
            is_txt_sti=is_txt_sti,
            trigger_type=trigger_type)


class CalibrationDisplay(RSVPDisplay):
    """ Calibration object of RSVP
        Attr:
            information(visual_Text_Stimuli): information text.
            task(visual_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock,
                 experiment_clock,
                 marker_writer,
                 text_info='Press Space Bar to Pause',
                 color_info='White', pos_info=(0, -.9),
                 height_info=0.2, font_info='Times',
                 color_task=['white'],
                 font_task='Times', text_task='1/100', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=True,
                 trigger_type='image'):
        """ Initializes Calibration Task Objects """

        color_text = [color_info]
        font_text = [font_info]
        text_text = [text_info]
        pos_text = [pos_info]
        height_text = [height_info]

        tmp = visual.TextStim(win=window, font=font_task, text=text_task)
        x_pos_task = tmp.boundingBox[0] / window.size[0] - 1
        pos_task = (x_pos_task, 1 - height_task)

        super(CalibrationDisplay, self).__init__(
            window, clock,
            experiment_clock,
            marker_writer,
            color_task=color_task,
            font_task=font_task,
            pos_task=pos_task,
            task_height=height_task,
            text_task=text_task,
            color_text=color_text,
            text_text=text_text,
            font_text=font_text,
            pos_text=pos_text,
            height_text=height_text,
            font_sti=font_sti,
            pos_sti=pos_sti,
            sti_height=sti_height,
            stim_sequence=stim_sequence,
            color_list_sti=color_list_sti,
            time_list_sti=time_list_sti,
            tr_pos_bg=tr_pos_bg,
            bl_pos_bg=bl_pos_bg,
            size_domain_bg=size_domain_bg,
            color_bg_txt=color_bg_txt,
            font_bg_txt=font_bg_txt,
            color_bar_bg=color_bar_bg,
            is_txt_sti=is_txt_sti,
            trigger_type=trigger_type)

class IconToIconDisplay(RSVPDisplay):
    """ Icon matching task object of RSVP
        Attr:
            information(visual_Text_Stimuli): information text.
            task(visual_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock,
                 experiment_clock,
                 marker_writer,
                 text_info='Press Space Bar to Pause',
                 color_info='White', pos_info=(0, -.9),
                 height_info=0.2, font_info='Times',
                 color_task=['white'],
                 font_task='Times', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=True,
                 trigger_type='image'):
        """ Initializes Icon Matching Task Objects """

        color_text = [color_info]
        font_text = [font_info]
        text_text = [text_info]
        pos_text = [pos_info]
        height_text = [height_info]

        tmp = visual.TextStim(win=window, font=font_task, text=' ')
        x_pos_task = tmp.boundingBox[0] / window.size[0] - 1
        self.pos_task = (x_pos_task, 1 - height_task)
        self.sti_height = sti_height
        self.pos_sti = pos_sti

        super(IconToIconDisplay, self).__init__(
            window, clock,
            experiment_clock,
            marker_writer,
            color_task=color_task,
            font_task=font_task,
            pos_task=self.pos_task,
            task_height=height_task,
            color_text=color_text,
            text_text=text_text,
            font_text=font_text,
            pos_text=pos_text,
            height_text=height_text,
            font_sti=font_sti,
            pos_sti=pos_sti,
            sti_height=sti_height,
            stim_sequence=stim_sequence,
            color_list_sti=color_list_sti,
            time_list_sti=time_list_sti,
            tr_pos_bg=tr_pos_bg,
            bl_pos_bg=bl_pos_bg,
            size_domain_bg=size_domain_bg,
            color_bg_txt=color_bg_txt,
            font_bg_txt=font_bg_txt,
            color_bar_bg=color_bar_bg,
            is_txt_sti=is_txt_sti,
            trigger_type=trigger_type)

        self.rect = visual.Rect(win=window, width=height_task, height=height_task, lineColor='black', pos=(pos_sti), lineWidth=10, ori=0.0)
        self.rect_drawn_frames = 0

        self.task = visual.ImageStim(win=window, image=None, mask=None,
                                    units='', pos=self.pos_task,
                                    size=(height_task * 2, height_task * 2),
                                    ori=0.0)

        self.target_text = visual.TextStim(win=window, color='yellow', text='TARGET:', pos=(pos_sti[0] - 0.5, pos_sti[1]), height=height_task)

    def draw_static(self):
        """Draw static elements in a stimulus."""
        if(self.rect_drawn_frames < self.time_to_present):
            self.rect.draw()
            self.target_text.draw()
            self.rect_drawn_frames += 1

        super(IconToIconDisplay, self).draw_static()

    def update_task_state(self, image_path, task_height, rect_color):
        """ Updates task state of Icon to Icon Matching Task by changing the
        image displayed at the top of the screen.
        Also updates rectangle size.
            Args:
                image_path: the path to the image to be displayed
                task_height: the height of the task image
                rect_color: the color of the rectangle"""

        self.task.image = image_path

        #Retrieve image width and height
        with Image.open(image_path) as pillow_image:
            image_width, image_height = pillow_image.size
        #Resize image so that its largest dimension is the stimuli size defined in the parameters file
        if image_width >= image_height:
            image_height = (image_height / image_width) * task_height
            image_width = task_height
        else:
            image_width = (image_width / image_height) * task_height
            image_height = task_height

        self.target_text.pos = (self.pos_sti[0] - image_width - 0.5, self.pos_sti[1])

        self.task.pos=(self.pos_task[0] + image_width * 2, self.pos_task[1] - image_width/2)
        self.task.size = (image_width * 2, image_height * 2)

        self.rect_drawn_frames = 0
        self.rect.width = image_width/task_height * self.sti_height
        self.rect.height = image_height/task_height * self.sti_height
        self.rect.lineColor = rect_color
