"""VEP Calibration task-related code"""
from typing import Any, Dict, Iterator, List, Optional

from psychopy import visual  # type: ignore

from bcipy.display import InformationProperties, VEPStimuliProperties
from bcipy.display.components.layout import centered
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.vep.codes import DEFAULT_FLICKER_RATES
from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.display.paradigm.vep.layout import BoxConfiguration
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.triggers import TriggerType
from bcipy.task.calibration import BaseCalibrationTask, Inquiry
from bcipy.task.paradigm.vep.stim_generation import \
    generate_vep_calibration_inquiries


class VEPCalibrationTask(BaseCalibrationTask):
    """VEP Calibration Task.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> highlighting inquiries -->
        saving data

    PARAMETERS:
    ----------
    parameters (dict)
    file_save (str)
    fake (bool)
    """
    name = 'VEP Calibration'
    paradigm = 'VEP'

    def __init__(self,
                 parameters: Parameters,
                 file_save: str,
                 fake: bool = False,
                 **kwargs: Any) -> None:
        self.box_colors = [
            '#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232'
        ]
        self.num_boxes = 6
        super().__init__(parameters, file_save, fake=fake, **kwargs)

    def init_display(self) -> VEPDisplay:
        """Initialize the display"""
        return init_vep_display(self.parameters, self.window,
                                self.experiment_clock, self.symbol_set,
                                self.box_colors)

    def init_inquiry_generator(self) -> Iterator[Inquiry]:
        """Initializes a generator that returns inquiries to be presented."""
        parameters = self.parameters
        schedule = generate_vep_calibration_inquiries(
            alp=self.symbol_set,
            timing=[
                parameters['time_prompt'], parameters['time_fixation'],
                parameters['time_flash']
            ],
            color=[
                parameters['target_color'], parameters['fixation_color'],
                *self.box_colors
            ],
            inquiry_count=parameters['stim_number'],
            num_boxes=self.num_boxes)
        return (Inquiry(*inq) for inq in schedule.inquiries())

    def trigger_type(self, symbol: str, target: str,
                     index: int) -> TriggerType:
        if target == symbol:
            return TriggerType.TARGET
        return TriggerType.EVENT

    def session_task_data(self) -> Dict[str, Any]:
        """Task-specific session data"""
        assert isinstance(self.display, VEPDisplay)
        boxes = [{
            "colors": box.colors,
            "flicker_rate": self.display.flicker_rates[i],
            "envelope": box.bounds
        } for i, box in enumerate(self.display.vep)]
        return {
            "boxes": boxes,
            "symbol_starting_positions": self.display.starting_positions
        }

    def session_inquiry_data(self,
                             inquiry: Inquiry) -> Optional[Dict[str, Any]]:
        """Defines task-specific session data for each inquiry."""
        assert isinstance(self.display, VEPDisplay)
        target_box = target_box_index(inquiry)
        target_freq = self.display.flicker_rates[
            target_box] if target_box is not None else None
        return {
            'target_box_index': target_box,
            'target_frequency': target_freq
        }

    def stim_labels(self, inquiry: Inquiry) -> List[str]:
        """labels for each stimuli in the session data."""
        target_box = target_box_index(inquiry)
        targetness = [TriggerType.NONTARGET for _ in range(self.num_boxes)]
        if target_box is not None:
            targetness[target_box] = TriggerType.TARGET
        labels = [TriggerType.PROMPT, TriggerType.FIXATION, *targetness]
        return list(map(str, labels))


def target_box_index(inquiry: Inquiry) -> Optional[int]:
    """Index of the target box."""
    target_letter, _fixation, *boxes = inquiry.stimuli
    for i, box in enumerate(boxes):
        if target_letter in box:
            return i
    return None


def init_vep_display(parameters: Parameters, window: visual.Window,
                     experiment_clock: Clock, symbol_set: List[str],
                     box_colors: List[str]) -> VEPDisplay:
    """Initialize the display"""
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )

    layout = centered(width_pct=0.95, height_pct=0.80)
    box_config = BoxConfiguration(layout, height_pct=0.30)

    timing = [
        parameters['time_prompt'], parameters['time_fixation'],
        parameters['time_flash']
    ]
    colors = [
        parameters['target_color'], parameters['fixation_color'], *box_colors
    ]
    stim_props = VEPStimuliProperties(
        stim_font=parameters['font'],
        stim_pos=box_config.positions,
        stim_height=0.1,
        timing=timing,
        stim_color=colors,
        inquiry=[],
        stim_length=1,
        animation_seconds=parameters['time_vep_animation'])

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height'])

    # issue #186641183 ; determine a better configuration strategy for flicker

    return VEPDisplay(window,
                      experiment_clock,
                      stim_props,
                      task_bar,
                      info,
                      symbol_set=symbol_set,
                      box_config=box_config,
                      flicker_rates=DEFAULT_FLICKER_RATES,
                      should_prompt_target=True)
