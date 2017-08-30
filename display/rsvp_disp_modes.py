#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
from psychopy import visual, core, event
from rsvp_disp import DisplayRSVP

""" RSVP Tasks are DisplayRSVP objects with different structure. They share
    the tasks and the essential elements and stimuli. However layout, length of
    stimuli list, update procedures and colors are different. Therefore each
    mode should be separated from each other carefully.
    Functions:
        update_task_state: update task information of the module """


class CopyPhraseTask(DisplayRSVP):
    """ Copy Phrase Task object of RSVP
        Attr:
            static_task(visual_Text_Stimuli): aim string of the copy phrase.
                (Stored in self.text[0])
            information(visual_Text_Stimuli): information text. (Stored in
                self.text[1])
            task(Multicolor_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock, static_text_task='COPY_PHRASE',
                 static_color_task='White',
                 text_information='Press Space Bar to Pause',
                 color_information='White', pos_information=(0, -.9),
                 height_information=0.2, font_information='Times',
                 color_task=['white'] * 4 + ['green'] * 2 + ['red'],
                 font_task='Times', text_task='COPY_PH', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 ele_list_sti=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=1):
        """ Initializes Copy Phrase Task Objects """

        tmp = visual.TextStim(window, font=font_task, text=static_text_task)
        static_pos_task = (
            tmp.boundingBox[0] / window.size[0] - 1, 1 - height_task)

        color_text = [static_color_task, color_information]
        font_text = [font_task, font_information]
        text_text = [static_text_task, text_information]
        pos_text = [static_pos_task, pos_information]
        height_text = [height_task, height_information]

        # Adjust task position wrt. static task position. Definition of
        # dummy texts are required. Place the task on bottom
        tmp2 = visual.TextStim(window, font=font_task, text=text_task)
        x_pos_task = tmp2.boundingBox[0] / window.size[0] - 1
        pos_task = (x_pos_task, static_pos_task[1] - height_task)

        super(CopyPhraseTask, self).__init__(window, clock,
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
                                             ele_list_sti=ele_list_sti,
                                             color_list_sti=color_list_sti,
                                             time_list_sti=time_list_sti,
                                             tr_pos_bg=tr_pos_bg,
                                             bl_pos_bg=bl_pos_bg,
                                             size_domain_bg=size_domain_bg,
                                             color_bg_txt=color_bg_txt,
                                             font_bg_txt=font_bg_txt,
                                             color_bar_bg=color_bar_bg,
                                             is_txt_sti=is_txt_sti)

    def update_task_state(self, text, color_list):
        """ Updates task state of Copy Phrase Task by removing letters or
            appending to the right.
            Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each """
        tmp2 = visual.TextStim(self.win, font=self.task.font, text=text)
        x_pos_task = tmp2.boundingBox[0] / self.win.size[0] - 1
        pos_task = (x_pos_task, self.text[0].pos[1] - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=pos_task)


class FreeSpellingTask(DisplayRSVP):
    """ Free Spelling Task object of RSVP
        Attr:
            information(visual_Text_Stimuli): information text.
            task(visual_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock,
                 text_information='Press Space Bar to Pause',
                 color_information='White', pos_information=(0, -.9),
                 height_information=0.2, font_information='Times',
                 color_task=['white'],
                 font_task='Times', text_task='1/100', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 ele_list_sti=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=1):
        """ Initializes Free Spelling Task Objects """

        color_text = [color_information]
        font_text = [font_information]
        text_text = [text_information]
        pos_text = [pos_information]
        height_text = [height_information]

        tmp = visual.TextStim(window, font=font_task, text=text_task)
        x_pos_task = tmp.boundingBox[0] / window.size[0] - 1
        pos_task = (x_pos_task, 1 - height_task)

        super(FreeSpellingTask, self).__init__(window, clock,
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
                                               ele_list_sti=ele_list_sti,
                                               color_list_sti=color_list_sti,
                                               time_list_sti=time_list_sti,
                                               tr_pos_bg=tr_pos_bg,
                                               bl_pos_bg=bl_pos_bg,
                                               size_domain_bg=size_domain_bg,
                                               color_bg_txt=color_bg_txt,
                                               font_bg_txt=font_bg_txt,
                                               color_bar_bg=color_bar_bg,
                                               is_txt_sti=is_txt_sti)

    def update_task_state(self, text, color_list):
        """ Updates task state of Free Spelling Task by removing letters or
            appending to the right.
            Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each """
        tmp = visual.TextStim(self.win, font=self.task.font, text=text)
        x_pos_task = tmp.boundingBox[0] / self.win.size[0] - 1
        pos_task = (x_pos_task, 1 - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=pos_task)


class CalibrationTask(DisplayRSVP):
    """ Calibration object of RSVP
        Attr:
            information(visual_Text_Stimuli): information text.
            task(visual_Text_Stimuli): task visualization.
            sti(visual_Text_Stimuli): stimuli text
            bg(BarGraph): bar graph display unit in display """

    def __init__(self, window, clock,
                 text_information='Press Space Bar to Pause',
                 color_information='White', pos_information=(0, -.9),
                 height_information=0.2, font_information='Times',
                 color_task=['white'],
                 font_task='Times', text_task='1/100', height_task=0.1,
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 ele_list_sti=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 is_txt_sti=1):
        """ Initializes Calibration Task Objects """

        color_text = [color_information]
        font_text = [font_information]
        text_text = [text_information]
        pos_text = [pos_information]
        height_text = [height_information]

        tmp = visual.TextStim(window, font=font_task, text=text_task)
        x_pos_task = tmp.boundingBox[0] / window.size[0] - 1
        pos_task = (x_pos_task, 1 - height_task)

        super(CalibrationTask, self).__init__(window, clock,
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
                                              ele_list_sti=ele_list_sti,
                                              color_list_sti=color_list_sti,
                                              time_list_sti=time_list_sti,
                                              tr_pos_bg=tr_pos_bg,
                                              bl_pos_bg=bl_pos_bg,
                                              size_domain_bg=size_domain_bg,
                                              color_bg_txt=color_bg_txt,
                                              font_bg_txt=font_bg_txt,
                                              color_bar_bg=color_bar_bg,
                                              is_txt_sti=is_txt_sti)

    def update_task_state(self, text, color_list):
        """ Updates task state of Calibration Task by removing letters or
            appending to the right.
            Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each """
        tmp = visual.TextStim(self.win, font=self.task.font, text=text)
        x_pos_task = tmp.boundingBox[0] / self.win.size[0] - 1
        pos_task = (x_pos_task, 1 - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=pos_task)
