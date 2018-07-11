# -*- coding: utf-8 -*-
from psychopy import visual, core
from PIL import Image

from bcipy.helpers.triggers import _calibration_trigger
from bcipy.display.display_main import BarGraph, MultiColorText
from bcipy.acquisition.marker_writer import NullMarkerWriter


class RSVPDisplay(object):
    """RSVP Display Object for Sequence Presentation.

    Animates a sequence in RSVP. Mode should be determined outside.
    """

    def __init__(self, window, static_period, experiment_clock,
                 marker_writer=None,
                 color_task=['white'],
                 font_task='Times',
                 pos_task=(-.8, .9), task_height=0.2, text_task='1/100',
                 color_text=['white'], text_text=['Information Text'],
                 font_text=['Times'], pos_text=[(.8, .9)], height_text=[0.2],
                 font_sti='Times', pos_sti=(-.8, .9), sti_height=0.2,
                 stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
                 time_list_sti=[1] * 10,
                 tr_pos_bg=(.5, .5), bl_pos_bg=(-.5, -.5), size_domain_bg=7,
                 color_bg_txt='red', font_bg_txt='Times', color_bar_bg='green',
                 bg_step_num=20, is_txt_sti=True,
                 static_period_time=.05,
                 trigger_type='image', bg=False):
        """Initialize RSVP window parameters and objects.

        Args:
                window(visual_window): Window in computer
                marker_writer(MarkerWriter): object used to write triggers to
                    the daq stream.
                color_task(list[string]): Color of the task string. Shares the
                    length of the text_task. If of length 1 the entire task
                    bar shares the same color.
                font_task(string): Font of task string
                pos_task(tuple): position of task string
                task_height(float): height for task string
                text_task(string): text of the task bar

                text_text(list[string]): text list for information texts
                color_text(list[string]): Color of the text string
                font_text(list[string]): Font of text string
                pos_text(list[tuple]): position of text string
                task_height(list[float]): height for text string

                sti_height(float): height of the stimuli object
                pos_sti(tuple): position of stimuli
                font_sti(string): font of the stimuli
                stim_sequence(list[string]): list of elements to flash
                color_list_sti(list[string]): list of colors for stimuli
                time_list_sti(list[float]): timing for each letter flash

                tr_pos_bg(tuple): top right corner location of bar graph
                bl_pos_bg(tuple): bottom left corner location of bar graph
                size_domain_bg(int): number of elements in bar graph
                color_bg_txt(string): color of bar graph text
                font_bg_txt(string): font of bar graph text
                color_bar_bg(string): color of bar graph bars
                bg_step_num(int): number of animation iterations for bars
        """
        self.win = window
        self.refresh_rate = window.getActualFrameRate()

        self.stim_sequence = stim_sequence
        self.color_list_sti = color_list_sti
        self.time_list_sti = time_list_sti

        self.is_txt_sti = is_txt_sti

        self.staticPeriod = static_period
        self.static_period_time = static_period_time
        self.experiment_clock = experiment_clock
        self.timing_clock = core.Clock()

        # Used to handle writing the marker stimulus
        self.marker_writer = marker_writer or NullMarkerWriter()

        # Length of the stimuli (number of flashes)
        self.len_sti = len(stim_sequence)

        # Stim parameters
        self.font_stim = font_sti
        self.height_stim = sti_height
        self.pos_sti = pos_sti

        self.first_run = True
        self.trigger_type = trigger_type

        # Check if task text is multicolored
        if len(color_task) == 1:
            self.task = visual.TextStim(win=window, color=color_task[0],
                                        height=task_height,
                                        text=text_task,
                                        font=font_task, pos=pos_task,
                                        wrapWidth=None, colorSpace='rgb',
                                        opacity=1, depth=-6.0)
        else:
            self.task = MultiColorText(win=window, list_color=color_task,
                                       height=task_height,
                                       text=text_task,
                                       font=font_task, pos=pos_task,
                                       opacity=1, depth=-6.0)

        # Create multiple text objects based on input
        self.text = []
        for idx in range(len(text_text)):
            self.text.append(visual.TextStim(win=window, color=color_text[idx],
                                             height=height_text[idx],
                                             text=text_text[idx],
                                             font=font_text[idx],
                                             pos=pos_text[idx],
                                             wrapWidth=None, colorSpace='rgb',
                                             opacity=1, depth=-6.0))

        # Create Stimuli Object
        if self.is_txt_sti:
            self.sti = visual.TextStim(win=window, color='white',
                                       height=sti_height, text='+',
                                       font=font_sti, pos=pos_sti,
                                       wrapWidth=None, colorSpace='rgb',
                                       opacity=1, depth=-6.0)
        else:
            self.sti = visual.ImageStim(win=window, image=None, mask=None,
                                        units='', pos=pos_sti,
                                        size=(sti_height, sti_height), ori=0.0)

        if bg:
            # Create Bar Graph
            self.bg = BarGraph(win=window, tr_pos_bg=tr_pos_bg,
                               bl_pos_bg=bl_pos_bg,
                               size_domain=size_domain_bg,
                               color_txt=color_bg_txt, font_bg=font_bg_txt,
                               color_bar_bg=color_bar_bg, max_num_step=bg_step_num)

    def draw_static(self):
        """Draw static elements in a stimulus."""
        self.task.draw()
        for idx in range(len(self.text)):
            self.text[idx].draw()

    def schedule_to(self, ele_list=[], time_list=[], color_list=[]):
        """Schedule stimuli elements (works as a buffer).

        Args:
                ele_list(list[string]): list of elements of stimuli
                time_list(list[float]): list of timings of stimuli
                color_list(list[string]): colors of elements of stimuli
        """
        self.stim_sequence = ele_list
        self.time_list_sti = time_list
        self.color_list_sti = color_list

    def update_task(self, text, color_list, pos):
        """Update Task Object.

        Args:
                text(string): text for task
                color_list(list[string]): list of the colors for each char
                pos(tuple): position of task
        """
        if len(color_list) == 1:
            self.task.text = text
            self.task.color = color_list[0]
            self.task.pos = pos
        else:
            self.task.update(text=text, color_list=color_list, pos=pos)

    def do_sequence(self):
        """Animate a sequence."""

        # init an array for timing information
        timing = []

        if self.first_run:
            # Play a sequence start sound to help orient triggers
            stim_timing = _calibration_trigger(
                self.experiment_clock,
                trigger_type=self.trigger_type, display=self.win,
                on_trigger=self.marker_writer.push_marker)

            timing.append(stim_timing)

            self.first_run = False

        # Do the sequence
        for idx in range(len(self.stim_sequence)):

            # Set a static period to do all our stim setting.
            #   Will warn if ISI value is violated.
            self.staticPeriod.start(self.static_period_time)

            # Turn ms timing into frames! Much more accurate!
            time_to_present = int(self.time_list_sti[idx] * self.refresh_rate)

            # Set the Stimuli attrs
            if self.is_txt_sti:
                self.sti.text = self.stim_sequence[idx]
                self.sti.color = self.color_list_sti[idx]
                sti_label = self.sti.text
            else:
                self.sti.image = self.stim_sequence[idx]
                
                #Retrieve image width and height
                with Image.open(self.sti.image) as pillow_image:
                    image_width, image_height = pillow_image.size
                #Resize image so that its largest dimension is the stimuli size defined in the parameters file
                if image_width >= image_height:
                    self.sti.size = (self.height_stim, (image_height / image_width) * self.height_stim)
                else:
                    self.sti.size = ((image_width / image_height) * self.height_stim, self.height_stim)
                    
                # We expect a path for images, so split on forward slash and
                # extension to get the name of the file.
                sti_label = self.sti.image.split('/')[-1].split('.')[0]

            # End static period
            self.staticPeriod.complete()

            # Reset the timing clock to start presenting
            self.timing_clock.reset()

            # Push to configured marker stream (ex. LslMarkerWriter)
            self.marker_writer.push_marker(sti_label)

            # Draw stimulus for n frames
            for n_frames in range(time_to_present):
                self.sti.draw()
                self.draw_static()
                self.win.flip()

            # Get trigger time (takes < .01ms)
            trigger_time = self.experiment_clock.getTime() - self.timing_clock.getTime()

            # Start another ISI for trigger saving
            self.staticPeriod.start(self.static_period_time)

            # append timing information
            if self.is_txt_sti:
                timing.append((sti_label, (trigger_time)))
            else:
                timing.append((sti_label, trigger_time))

            # End the static period
            self.staticPeriod.complete()

        # draw in static and flip once more
        self.draw_static()
        self.win.flip()

        return timing

    def show_bar_graph(self):
        """Show Bar Graph."""

        for idx in range(self.bg.max_num_step):
            self.draw_static()
            self.bg.animate(idx)
            self.win.flip()

    def update_task_state(self, text, color_list):
        """Update task state.

        Removes letters or appends to the right.
        Args:
                text(string): new text for task state
                color_list(list[string]): list of colors for each
        """
        tmp = visual.TextStim(win=self.win, font=self.task.font, text=text)
        x_pos_task = tmp.boundingBox[0] / self.win.size[0] - 1
        pos_task = (x_pos_task, 1 - self.task.height)

        self.update_task(text=text, color_list=color_list, pos=pos_task)

    def wait_screen(self, message, color):
        """Wait Screen.

        Args:
            message(string): message to be displayed while waiting
        """

        # Construct the wait message
        wait_message = visual.TextStim(win=self.win, font=self.font_stim,
                                       text=message,
                                       height=.1,
                                       color=color,
                                       pos=(0, -.5),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1, depth=-6.0)

        # Try adding our BCI logo. Pass if not found.
        try:
            wait_logo = visual.ImageStim(
                self.win,
                image='bcipy/static/images/gui_images/bci_cas_logo.png',
                size=(1, 1),
                pos=(0, .5),
                mask=None,
                ori=0.0)
            wait_logo.draw()

        except Exception:
            print("Cannot load logo image")
            pass

        # Draw and flip the screen.
        wait_message.draw()
        self.win.flip()
