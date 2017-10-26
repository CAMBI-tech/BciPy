# Calibration Task for RSVP

from __future__ import division
from psychopy import visual, core

from display.rsvp_disp_modes import CalibrationTask
from utils.trigger_helpers import _write_triggers_from_sequence_calibration


def RSVP_calibration_task(win, daq, parameters, file_save):
    daq.start_acquistion(file_save)
    # Initialize Experiment clocks etc.
    frame_rate = win.getActualFrameRate()
    clock = core.StaticPeriod(screenHz=frame_rate)
    experiment_clock = core.MonotonicClock(start_time=None)

    rsvp = CalibrationTask(
        window=win, clock=clock,
        experiment_clock=experiment_clock,
        text_information=parameters['text_text']['value'],
        color_information=parameters['color_text']['value'],
        pos_information=parameters['pos_text']['value'],
        height_information=parameters['txt_height']['value'],
        font_information=parameters['font_text']['value'],
        color_task=['white'],
        font_task=parameters['font_task']['value'],
        text_task=task_text[0],
        height_task=parameters['height_task']['value'],
        font_sti=parameters['font_sti']['value'],
        pos_sti=parameters['pos_sti']['value'],
        sti_height=parameters['sti_height']['value'],
        ele_list_sti=['a'] * 10, color_list_sti=['white'] * 10,
        time_list_sti=[3] * 10,
        tr_pos_bg=parameters['tr_pos_bg']['value'],
        bl_pos_bg=parameters['bl_pos_bg']['value'],
        size_domain_bg=parameters['size_domain_bg']['value'],
        color_bg_txt=parameters['color_bg_txt']['value'],
        font_bg_txt=parameters['font_bg_txt']['value'],
        color_bar_bg=parameters['color_bar_bg']['value'],
        is_txt_sti=parameters['is_txt_sti']['value'])

    # Init Task
    run = True
    while run is True:
        # to-do allow pausing and exiting. See psychopy getKeys()
        (task_text, task_color,
            task_timing, ele_sti, timing_sti, color_sti) = get_task_info()

        for idx_o in range(len(task_text)):

            rsvp.update_task_state(
                text=task_text[idx_o],
                color_list=task_color[idx_o])
            rsvp.draw_static()
            win.flip()
            rsvp.sti.height = parameters['sti_height']['value']

            # Schedule a sequence
            rsvp.ele_list_sti = ele_sti[idx_o]

            if parameters['is_txt_sti']['value']:
                rsvp.color_list_sti = color_sti[idx_o]

            rsvp.time_list_sti = timing_sti[idx_o]

            core.wait(.4)
            sequence_timing = rsvp.do_sequence()

            # _write_triggers_from_sequence_calibration(sequence_timing, file)

            core.wait(.5)

        run = False
    daq.stop_acquistion()

    # close the window and file
    win.close()
    file.close()

    return (daq, file)
