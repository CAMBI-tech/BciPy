"""RSVP calibration task module.

This module provides the RSVP (Rapid Serial Visual Presentation) calibration task
implementation which performs stimulus inquiries to elicit ERPs. The task presents
stimuli in rapid succession with configurable timing and appearance parameters.
"""

from psychopy import core, visual

from bcipy.core.parameters import Parameters
from bcipy.display import Display, InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.task.calibration import BaseCalibrationTask


class RSVPCalibrationTask(BaseCalibrationTask):
    """RSVP Calibration Task.

    This task performs RSVP stimulus inquiries to elicit ERPs by presenting
    stimuli in rapid succession. Parameters control the number of stimuli,
    presentation duration, colors, and text/image inputs.

    Task flow:
        1. Setup variables
        2. Initialize EEG
        3. Await user input
        4. Setup stimuli
        5. Present inquiries
        6. Save data

    Attributes:
        name: Name of the task.
        paradigm: Name of the paradigm.
        parameters: Task configuration parameters.
        file_save: Path for saving task data.
        fake: Whether to run in fake (testing) mode.
        window: PsychoPy window for display.
        static_clock: Clock for static timing.
        experiment_clock: Clock for experiment timing.
    """

    name = 'RSVP Calibration'
    paradigm = 'RSVP'

    def init_display(self) -> Display:
        """Initialize the RSVP display.

        Returns:
            Display: Configured RSVP calibration display instance.
        """
        return init_calibration_display_task(self.parameters, self.window,
                                             self.static_clock,
                                             self.experiment_clock)


def init_calibration_display_task(
        parameters: Parameters, window: visual.Window,
        static_clock: core.StaticPeriod,
        experiment_clock: Clock) -> CalibrationDisplay:
    """Initialize the RSVP calibration display.

    Args:
        parameters: Task configuration parameters.
        window: PsychoPy window for display.
        static_clock: Clock for static timing.
        experiment_clock: Clock for experiment timing.

    Returns:
        CalibrationDisplay: Configured RSVP calibration display instance.
    """
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
