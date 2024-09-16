from psychopy import core, visual

from bcipy.display import (Display, InformationProperties,
                           PreviewInquiryProperties, StimuliProperties)
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.triggers import TriggerType
from bcipy.task.base_calibration import BaseCalibrationTask


class RSVPCalibrationTask(BaseCalibrationTask):
    """RSVP Calibration Task.

    Calibration task performs an RSVP stimulus inquiry
        to elicit an ERP. Parameters will change how many stimuli
        and for how long they present. Parameters also change
        color and text / image inputs.

    This task progresses as follows:

    setting up variables --> initializing eeg --> awaiting user input to start --> setting up stimuli -->
    presenting inquiries --> saving data

    PARAMETERS:
    ----------
    win (PsychoPy Display)
    daq (Data Acquisition Client)
    parameters (dict)
    file_save (str)
    """
    MODE = 'RSVP'

    def trigger_type(self, symbol: str, target: str,
                     index: int) -> TriggerType:
        if index == 0:
            return TriggerType.PROMPT
        if symbol == 'inquiry_preview':
            return TriggerType.PREVIEW
        if symbol == '+':
            return TriggerType.FIXATION
        if target == symbol:
            return TriggerType.TARGET
        return TriggerType.NONTARGET

    def init_display(self) -> Display:
        return init_calibration_display_task(self.parameters, self.window,
                                             self.static_clock,
                                             self.experiment_clock)


def init_calibration_display_task(
        parameters: Parameters, window: visual.Window,
        static_clock: core.StaticPeriod,
        experiment_clock: Clock) -> CalibrationDisplay:
    """Initialize the display"""
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(
        stim_font=parameters['font'],
        stim_pos=(parameters['stim_pos_x'], parameters['stim_pos_y']),
        stim_height=parameters['stim_height'],
        stim_inquiry=[''] * parameters['stim_length'],
        stim_colors=[parameters['stim_color']] * parameters['stim_length'],
        stim_timing=[10] * parameters['stim_length'],
        is_txt_stim=parameters['is_txt_stim'])

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height'],
                                  padding=parameters['task_padding'])

    preview_inquiry = PreviewInquiryProperties(
        preview_on=parameters['show_preview_inquiry'],
        preview_only=True,
        preview_inquiry_length=parameters['preview_inquiry_length'],
        preview_inquiry_progress_method=parameters[
            'preview_inquiry_progress_method'],
        preview_inquiry_key_input=parameters['preview_inquiry_key_input'],
        preview_inquiry_isi=parameters['preview_inquiry_isi'])

    return CalibrationDisplay(window,
                              static_clock,
                              experiment_clock,
                              stimuli,
                              task_bar,
                              info,
                              preview_inquiry=preview_inquiry,
                              trigger_type=parameters['trigger_type'],
                              space_char=parameters['stim_space_char'],
                              full_screen=parameters['full_screen'])
