"""VEP Calibration task-related code"""
from typing import Any, Dict, List, Tuple, Union

from psychopy import core, visual  # type: ignore

from bcipy.acquisition.multimodal import ClientManager
from bcipy.config import (SESSION_DATA_FILENAME, TRIGGER_FILENAME,
                          WAIT_SCREEN_MESSAGE)
from bcipy.display import InformationProperties, VEPStimuliProperties
from bcipy.display.components.layout import centered
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.vep.codes import DEFAULT_FLICKER_RATES
from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.display.paradigm.vep.layout import BoxConfiguration
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.task import get_user_input, trial_complete_message
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, convert_timing_triggers,
                                    offset_label)
from bcipy.task import Task
from bcipy.task.data import Inquiry, Session
from bcipy.task.paradigm.vep.stim_generation import (
    InquirySchedule, generate_vep_calibration_inquiries)


def trigger_type(symbol: str, target: str, _index: int) -> TriggerType:
    """Trigger Type.

    This method is passed to convert_timing_triggers to properly assign TriggerTypes
        to the timing of stimuli presented.
    """
    if target == symbol:
        return TriggerType.TARGET
    return TriggerType.EVENT


class VEPCalibrationTask(Task):
    """VEP Calibration Task.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> highlighting inquiries -->
        saving data

    PARAMETERS:
    ----------
    win (PsychoPy Display Object)
    daq (Data Acquisition Object)
    parameters (Dictionary)
    file_save (String)
    """
    MODE = 'VEP'

    def __init__(self, win: visual.Window, daq: ClientManager,
                 parameters: Parameters, file_save: str):
        super(VEPCalibrationTask, self).__init__()
        self.num_boxes = 6
        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = Clock()
        self.start_time = self.experiment_clock.getTime()
        self.buffer_val = parameters['task_buffer_length']
        self.symbol_set = alphabet(parameters)

        self.file_save = file_save
        self.trigger_handler = TriggerHandler(self.file_save, TRIGGER_FILENAME,
                                              FlushFrequency.EVERY)

        self.wait_screen_message = WAIT_SCREEN_MESSAGE
        self.wait_screen_message_color = parameters['stim_color']

        self.stim_number = parameters['stim_number']

        self.timing = [
            parameters['time_prompt'], parameters['time_fixation'],
            parameters['time_flash']
        ]

        self.color = [
            parameters['target_color'], parameters['fixation_color'],
            '#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232'
        ]

        self.is_txt_stim = parameters['is_txt_stim']
        self.display = self.init_display()

        self.session = Session(save_location=self.file_save,
                               task='Calibration',
                               mode=self.MODE,
                               symbol_set=self.symbol_set,
                               task_data=self.session_task_data())
        self.write_session_data()

    def session_task_data(self) -> Dict[str, Any]:
        """Task-specific session data"""
        boxes = [{
            "colors": box.colors,
            "flicker_rate": self.display.flicker_rates[i],
            "envelope": box.bounds
        } for i, box in enumerate(self.display.vep)]
        return {
            "boxes": boxes,
            "symbol_starting_positions": self.display.starting_positions
        }

    def init_display(self) -> VEPDisplay:
        """Initialize the display"""
        return init_calibration_display(self.parameters, self.window,
                                        self.experiment_clock, self.symbol_set,
                                        self.timing, self.color)

    def generate_stimuli(self) -> InquirySchedule:
        """Generates the inquiries to be presented.
        Returns:
        --------
            tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)
        """
        return generate_vep_calibration_inquiries(
            alp=self.symbol_set,
            timing=self.timing,
            color=self.color,
            inquiry_count=self.stim_number,
            num_boxes=self.num_boxes,
            is_txt=self.is_txt_stim)

    def execute(self):
        """Main task loop"""
        self.logger.info(f'Starting {self.name()}!')
        run = True

        # Check user input to make sure we should be going
        if not get_user_input(self.display,
                              self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        # Wait for a time
        core.wait(self.buffer_val)

        # Begin the Experiment
        while run:

            # Get inquiry information given stimuli parameters
            (stimuli_labels, _timing, _colors) = self.generate_stimuli()

            for inquiry in range(self.stim_number):
                # check user input to make sure we should be going
                if not get_user_input(self.display, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break

                # update task state
                self.display.update_task_bar(str(inquiry + 1))

                # Draw and flip screen
                self.display.draw_static()
                self.window.flip()

                self.display.schedule_to(stimuli_labels[inquiry])
                # Schedule an inquiry

                # Wait for a time
                core.wait(self.buffer_val)

                # Do the inquiry
                timing = self.display.do_inquiry()

                # Write triggers for the inquiry
                self.write_trigger_data(timing, first_run=(inquiry == 0))

                self.update_session_data(stimuli_labels[inquiry])

                # Wait for a time
                core.wait(self.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.display.info_text = trial_complete_message(
            self.window, self.parameters)
        self.display.draw_static()
        self.window.flip()

        # Allow for some training data to be collected
        core.wait(self.buffer_val)
        self.write_offset_trigger()

        return self.file_save

    def write_trigger_data(self, timing: List[Tuple[str, float]],
                           first_run: bool) -> None:
        """Write Trigger Data.

        Using the timing provided from the display and calibration information
        from the data acquisition client, write trigger data in the correct
        format.
        """
        triggers = []
        if first_run:
            assert self.display.first_stim_time, "First stim time not set"
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = client.offset(self.display.first_stim_time
                                     ) - self.display.first_stim_time
                triggers.append(Trigger(label, TriggerType.OFFSET, time))

        target = timing[0][0]
        triggers.extend(convert_timing_triggers(timing, target, trigger_type))
        self.trigger_handler.add_triggers(triggers)

    def write_offset_trigger(self) -> None:
        """Append an offset value to the end of the trigger file."""
        assert self.display.first_stim_time, "First stim time not set"
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix='daq_sample_offset')
            time = client.offset(self.display.first_stim_time)
            triggers.append(Trigger(label, TriggerType.SYSTEM, time))

        self.trigger_handler.add_triggers(triggers)
        self.trigger_handler.close()

    def write_session_data(self) -> None:
        """Save session data to disk."""
        if self.session:
            session_file = _save_session_related_data(
                f"{self.file_save}/{SESSION_DATA_FILENAME}",
                self.session.as_dict())
            session_file.close()

    def update_session_data(self, inquiry: List[Union[List[str],
                                                      str]]) -> None:
        """Adds the inquiry to the session data and writes to disk.

        Parameters
        ----------
        inquiry - list of target, fixation, and sublists representing each
            vep box and its content symbols.
        """
        target_letter, _fixation, *boxes = inquiry
        assert isinstance(target_letter, str), "unexpected inquiry format"

        target_box = None
        for i, box in enumerate(boxes):
            if target_letter in box:
                target_box = i

        target_freq = self.display.flicker_rates[
            target_box] if target_box is not None else None

        task_data = {
            'target_box_index': target_box,
            'target_frequency': target_freq
        }

        targetness = [TriggerType.NONTARGET for _ in range(self.num_boxes)]
        if target_box is not None:
            targetness[target_box] = TriggerType.TARGET
        trg_types = [TriggerType.PROMPT, TriggerType.FIXATION, *targetness]

        data = Inquiry(stimuli=inquiry,
                       timing=[],
                       triggers=[],
                       target_info=list(map(str, trg_types)),
                       target_letter=target_letter,
                       task_data=task_data)
        self.session.add_sequence(data)
        self.session.total_time_spent = self.experiment_clock.getTime(
        ) - self.start_time
        self.write_session_data()

    def name(self):
        return 'VEP Calibration Task'


def init_calibration_display(parameters: Parameters, window: visual.Window,
                             experiment_clock: Clock, symbol_set: List[str],
                             timing: List[float],
                             colors: List[str]) -> VEPDisplay:
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
