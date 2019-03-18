from psychopy import visual, clock
import numpy as np
import time
from bcipy.helpers.task import alphabet

import logging


class MatrixDisplay:
    """ Matrix Display Object for Sequence Presentation. Animates a sequence
        in Matrix. Mode should be determined outside.
        Attr:
            task(visual_Text_Stimuli): task bar
            text(list[visual_Text_Stimuli]): text bar, there can be
                different number of information texts in different paradigms
            sti(visual_Text_Stimuli): stimuli text
            trialClock(core_clock): timer for presentation """

    def __init__(self, window, clock, experiment_clock, task_color='white',
                 task_font='Times', task_pos=(-.75, .75), task_height=0.1,
                 grid_rows=6, grid_columns=6, time_flash=.25,
                 task_text='1/100',
                 stim_font='Times',
                 stim_pos=(0, .0),
                 stim_height=0.25,
                 is_txt_stim=True, alp=None):

        self.win = window
        self.logger = log = logging.getLogger(__name__)

        # TASK TEXT
        self.task_font_text = task_font
        self.task_height = task_height
        self.task_pos = task_pos
        self.task_text = task_text
        self.task_color = task_color

        # STIM / GRID
        self.is_txt_stim = is_txt_stim
        self.stimuli = []
        self.rows = grid_rows
        self.stimuli_font = stim_font
        self.columns = grid_columns
        self.stim_height = stim_height

        self.stim_number = grid_rows + grid_columns

        self.max_height_grid = -1 + stim_height
        self.max_width_grid = 1 - stim_height
        self.uniform_grid_values_row = sorted(
            np.linspace(
                start=self.max_height_grid,
                stop=self.max_width_grid,
                num=self.rows))

        self.uniform_grid_values_col = sorted(
            np.linspace(
                start=self.max_height_grid,
                stop=self.max_width_grid - .25,
                num=self.columns),
            reverse=True)

        if not alp:
            self.alp = alphabet()
        else:
            self.alp = alp

        self.flash = time_flash
        # Clocks
        self.trialClock = clock
        self.expClock = experiment_clock

        # Length of the stimuli (number of stimuli on screen)
        self.stim_length = len(self.stimuli)

    def draw_static(self):
        """ Draws static elements in a stimulus. """

        # Draw the grid
        if len(self.stimuli) is 0:
            self.make_spelling_grid()

        for stim in self.stimuli:
            stim.draw()

        # Next, Draw the task text
        task_text = visual.TextStim(win=self.win, color=self.task_color,
                                    height=self.task_height,
                                    text=self.task_text,
                                    font=self.task_font_text,
                                    pos=self.task_pos,
                                    wrapWidth=None, colorSpace='rgb',
                                    opacity=1, depth=-6.0)
        task_text.draw()

        self.win.flip()

    def make_spelling_grid(self):
        # First Draw the grid!
        alp_idx = 0

        # Loop through each row
        for idx in range(self.rows):
            if self.is_txt_stim:

                # loop through each column
                for x in range(self.columns):
                    if alp_idx > len(self.alp) - 1:
                        break

                    pos = determine_position_on_grid(
                        idx, x,
                        self.uniform_grid_values_col,
                        self.uniform_grid_values_row)

                    stim = visual.TextStim(win=self.win,
                                           height=self.stim_height,
                                           text=self.alp[alp_idx],
                                           font=self.stimuli_font, pos=pos,
                                           wrapWidth=None, colorSpace='rgb',
                                           opacity=1, depth=-6.0)
                    alp_idx += 1

                    stim.draw()
                    self.stimuli.append(stim)

    def do_sequence(self):
        """ implements grid sequence """

        # Stimulate rows
        for x in self.uniform_grid_values_row:
            for stim in self.stimuli:
                if stim.pos[0] == x:
                    stim.color = 'blue'
                else:
                    stim.color = 'white'

            self.draw_static()
            time.sleep(self.flash)

        # Stimulate columns
        for x in self.uniform_grid_values_col:
            for stim in self.stimuli:
                if stim.pos[1] == x:
                    stim.color = 'blue'
                else:
                    stim.color = 'white'

            self.draw_static()
            time.sleep(self.flash)


def determine_position_on_grid(row_idx, col_idx,
                               uniform_grid_values_row,
                               uniform_grid_values_col):
    # Position on screen goes from -1 to 1, where (0, 0) is the center of the
    #  screen. Assume an adequate size ratio of window.
    try:
        x = uniform_grid_values_col[col_idx]
        y = uniform_grid_values_row[row_idx]
    except Exception as e:
        log.debug(f'at index: {col_idx} Error: {e}')

    # # else:
    # x = 0

    return (x, y)

if __name__ == "__main__":
    # How many stimulations?
    task_length = 5

    # make the display window
    display_window = visual.Window(
        size=[1000,
              1000],
        screen=0,
        allowGUI=False,
        useFBO=False,
        fullscr=False,
        allowStencil=False,
        monitor='mainMonitor',
        winType='pyglet', units='norm', waitBlanking=True,
        color='black')

    # Create a Display Matrix Object
    matrix = MatrixDisplay(
        display_window,
        clock.MonotonicClock(),
        clock.MonotonicClock(),
        task_text='')

    # draw matrix grid and other static
    matrix.draw_static()

    task_index = 0
    for i in range(task_length):
        task_index += 1

        matrix.task_text = '%s/5' % task_index
        matrix.draw_static()

        # animate!
        matrix.do_sequence()

        time.sleep(2)
