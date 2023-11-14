"""VEP Calibration task-related code"""
from typing import List, Tuple

from psychopy import core, visual

from bcipy.acquisition.multimodal import ClientManager
from bcipy.config import TRIGGER_FILENAME, WAIT_SCREEN_MESSAGE
from bcipy.display import InformationProperties, VEPStimuliProperties
from bcipy.display.components.layout import centered
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.display.paradigm.vep.layout import BoxConfiguration
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import (DEFAULT_TEXT_FIXATION, InquirySchedule,
                                   generate_vep_calibration_inquiries)
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.task import get_user_input, trial_complete_message
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, convert_timing_triggers,
                                    offset_label)
from bcipy.task import Task


def trigger_type(symbol: str, target: str, index: int) -> TriggerType:
    """Trigger Type.

    This method is passed to convert_timing_triggers to properly assign TriggerTypes
        to the timing of stimuli presented.
    """
    if index == 0:
        return TriggerType.PROMPT
    if symbol == DEFAULT_TEXT_FIXATION:
        return TriggerType.FIXATION
    if target == symbol:
        return TriggerType.TARGET
    return TriggerType.NONTARGET


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
                # Schedule a inquiry

                # Wait for a time
                core.wait(self.buffer_val)

                # Do the inquiry
                timing = self.display.do_inquiry()

                # Write triggers for the inquiry
                self.write_trigger_data(timing, (inquiry == 0))

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

        *Note on offsets*: we write the full offset value which can be used to
        transform all stimuli to the time since session start (t = 0) for all
        values (as opposed to most system clocks which start much higher).
        We do not write the calibration trigger used to generate this offset
        from the display. See display _trigger_pulse() for more information.
        """
        if first_run:
            assert self.display.first_stim_time, "First stim time not set"
            triggers = []
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = client.offset(self.display.first_stim_time
                                     ) - self.display.first_stim_time
                triggers.append(Trigger(label, TriggerType.OFFSET, time))
            self.trigger_handler.add_triggers(triggers)

        # make sure triggers are written for the inquiry
        self.trigger_handler.add_triggers(
            convert_timing_triggers(timing, timing[0][0], trigger_type))

    def write_offset_trigger(self) -> None:
        """Append an offset value to the end of the trigger file.
        """
        assert self.display.first_stim_time, "First stim time not set"
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix='daq_sample_offset')
            time = client.offset(self.display.first_stim_time)
            triggers.append(Trigger(label, TriggerType.SYSTEM, time))

        self.trigger_handler.add_triggers(triggers)
        self.trigger_handler.close()

    def name(self):
        return 'VEP Calibration Task'


def init_calibration_display(parameters: Parameters,
                             window,
                             experiment_clock,
                             symbol_set,
                             timing,
                             colors,
                             num_boxes=6):
    """Initialize the display"""
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )

    layout = centered(width_pct=0.95, height_pct=0.80)
    box_config = BoxConfiguration(layout, num_boxes=num_boxes, height_pct=0.30)

    stim_props = VEPStimuliProperties(stim_font=parameters['font'],
                                      stim_pos=box_config.positions,
                                      stim_height=0.1,
                                      timing=timing,
                                      stim_color=colors,
                                      inquiry=[],
                                      stim_length=1,
                                      animation_seconds=1.0)

    task_bar = CalibrationTaskBar(window,
                                  inquiry_count=parameters['stim_number'],
                                  current_index=0,
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height'])

    return VEPDisplay(window,
                      experiment_clock,
                      stim_props,
                      task_bar,
                      info,
                      symbol_set=symbol_set,
                      box_config=box_config,
                      should_prompt_target=True)
