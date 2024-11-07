from psychopy import core, visual

from bcipy.display import Display, InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.data.parameters import Parameters
from bcipy.task.calibration import BaseCalibrationTask


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
    parameters (dict)
    file_save (str)
    fake (bool)
    """
    name = 'RSVP Calibration'
    paradigm = 'RSVP'

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
        stim_pos=(parameters['rsvp_stim_pos_x'], parameters['rsvp_stim_pos_y']),
        stim_height=parameters['rsvp_stim_height'],
        stim_inquiry=[''] * parameters['stim_length'],
        stim_colors=[parameters['stim_color']] * parameters['stim_length'],
        stim_timing=[10] * parameters['stim_length'],
        is_txt_stim=parameters['is_txt_stim'])

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['rsvp_task_height'],
                                  padding=parameters['rsvp_task_padding'])

    return CalibrationDisplay(window,
                              static_clock,
                              experiment_clock,
                              stimuli,
                              task_bar,
                              info,
                              preview_config=parameters.instantiate(PreviewParams),
                              trigger_type=parameters['trigger_type'],
                              space_char=parameters['stim_space_char'],
                              full_screen=parameters['full_screen'])
