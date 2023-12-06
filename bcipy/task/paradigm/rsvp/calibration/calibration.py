from typing import List, Tuple

from psychopy import core, visual

from bcipy.acquisition import ClientManager
from bcipy.config import TRIGGER_FILENAME, WAIT_SCREEN_MESSAGE
from bcipy.display import (InformationProperties, PreviewInquiryProperties,
                           StimuliProperties)
from bcipy.display.components.task_bar import CalibrationTaskBar
from bcipy.display.paradigm.rsvp.mode.calibration import CalibrationDisplay
from bcipy.helpers.clock import Clock
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import (InquirySchedule, StimuliOrder, TargetPositions,
                                   generate_calibration_inquiries)
from bcipy.helpers.symbols import alphabet
from bcipy.helpers.task import (get_user_input, pause_calibration,
                                trial_complete_message)
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, convert_timing_triggers,
                                    offset_label)

from bcipy.task import Task


class RSVPCalibrationTask(Task):
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

    def __init__(
            self,
            win: visual.Window,
            daq: ClientManager,
            parameters: Parameters,
            file_save: str) -> None:
        super(RSVPCalibrationTask, self).__init__()

        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = Clock()
        self.buffer_val = parameters['task_buffer_length']
        self.alp = alphabet(parameters)
        self.rsvp = init_calibration_display_task(
            self.parameters, self.window,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save
        self.trigger_handler = TriggerHandler(
            self.file_save,
            TRIGGER_FILENAME,
            FlushFrequency.EVERY)

        self.stim_number = parameters['stim_number']
        self.stim_length = parameters['stim_length']
        self.stim_order = StimuliOrder(parameters['stim_order'])
        self.target_positions = TargetPositions(parameters['target_positions'])
        self.nontarget_inquiries = parameters['nontarget_inquiries']

        self.show_preview_inquiry = parameters['show_preview_inquiry']

        self.timing = [parameters['time_prompt'],
                       parameters['time_fixation'],
                       parameters['time_flash']]
        self.jitter = parameters['stim_jitter']

        self.color = [parameters['target_color'],
                      parameters['fixation_color'],
                      parameters['stim_color']]
        self.wait_screen_message_color = self.color[-1]
        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['stim_height']
        self.is_txt_stim = parameters['is_txt_stim']
        self.enable_breaks = parameters['enable_breaks']

    def generate_stimuli(self) -> InquirySchedule:
        """Generates the inquiries to be presented.
        Returns:
        --------
            tuple(
                samples[list[list[str]]]: list of inquiries
                timing(list[list[float]]): list of timings
                color(list(list[str])): list of colors)
        """
        return generate_calibration_inquiries(self.alp,
                                              inquiry_count=self.stim_number,
                                              stim_per_inquiry=self.stim_length,
                                              stim_order=self.stim_order,
                                              target_positions=self.target_positions,
                                              percentage_without_target=self.nontarget_inquiries,
                                              timing=self.timing,
                                              jitter=self.jitter,
                                              is_txt=self.rsvp.is_txt_stim,
                                              color=self.color)

    def trigger_type(self, symbol: str, target: str, index: int) -> TriggerType:
        """Trigger Type.

        This method is passed to convert_timing_triggers to properly assign TriggerTypes
            to the timing of stimuli presented.
        """
        if index == 0:
            return TriggerType.PROMPT
        if symbol == 'inquiry_preview':
            return TriggerType.PREVIEW
        if symbol == '+':
            return TriggerType.FIXATION
        if target == symbol:
            return TriggerType.TARGET
        return TriggerType.NONTARGET

    def execute(self) -> str:

        self.logger.info(f'Starting {self.name()}!')
        run = True

        # Check user input to make sure we should be going
        if not get_user_input(self.rsvp, WAIT_SCREEN_MESSAGE,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        # Begin the Experiment
        while run:

            # Get inquiry information given stimuli parameters
            (stimuli, stimuli_timing, stimuli_color) = self.generate_stimuli()

            # Execute the RSVP inquiries
            for inquiry in range(self.stim_number):

                # check user input to make sure we should be going
                if not get_user_input(self.rsvp, WAIT_SCREEN_MESSAGE,
                                      self.wait_screen_message_color):
                    break

                # Take a break every number of trials defined
                if self.enable_breaks:
                    pause_calibration(self.window, self.rsvp, inquiry,
                                      self.parameters)

                # update task state
                self.rsvp.update_task_bar()

                # Draw and flip screen
                self.rsvp.draw_static()
                self.window.flip()

                # Schedule a inquiry
                self.rsvp.stimuli_inquiry = stimuli[inquiry]

                # check if text stimuli or not for color information
                if self.is_txt_stim:
                    self.rsvp.stimuli_colors = stimuli_color[inquiry]

                self.rsvp.stimuli_timing = stimuli_timing[inquiry]

                core.wait(self.buffer_val)

                # do inquiry and get timing information back from the display
                timing = self.rsvp.do_inquiry(preview_calibration=self.show_preview_inquiry)

                self.write_trigger_data(timing, (inquiry == 0))
                core.wait(self.buffer_val)

            # Set run to False to stop looping
            run = False

        # Say Goodbye!
        self.rsvp.info_text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        self.write_offset_trigger()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.buffer_val)

        return self.file_save

    def write_trigger_data(self, timing: List[Tuple[str, float]], first_run: bool) -> None:
        """Write Trigger Data.

        Using the timing provided from the display and calibration information from the data acquisition
        client, write trigger data in the correct format.

        *Note on offsets*: we write the full offset value which can be used to transform all stimuli to the time since
            session start (t = 0) for all values (as opposed to most system clocks which start much higher).
            We do not write the calibration trigger used to generate this offset from the display.
            See RSVPDisplay._trigger_pulse() for more information.
        """
        # write offsets. currently, we only check for offsets at the beginning.
        # offset will factor in true offset and time relative from beginning
        if first_run:
            triggers = []
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = client.offset(
                    self.rsvp.first_stim_time) - self.rsvp.first_stim_time
                triggers.append(Trigger(label, TriggerType.OFFSET, time))
            self.trigger_handler.add_triggers(triggers)

        # make sure triggers are written for the inquiry
        self.trigger_handler.add_triggers(convert_timing_triggers(timing, timing[0][0], self.trigger_type))

    def write_offset_trigger(self) -> None:
        """Append an offset value to the end of the trigger file.
        """
        # To help support future refactoring or use of lsl timestamps only
        # we write only the sample offset here.
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix='daq_sample_offset')
            time = client.offset(self.rsvp.first_stim_time)
            triggers.append(Trigger(label, TriggerType.SYSTEM, time))

        self.trigger_handler.add_triggers(triggers)
        self.trigger_handler.close()

    def name(self) -> str:
        return 'RSVP Calibration Task'


def init_calibration_display_task(
        parameters: Parameters,
        window: visual.Window,
        static_clock: core.StaticPeriod,
        experiment_clock: Clock) -> CalibrationDisplay:
    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'],
                   parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )
    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(parameters['stim_pos_x'],
                                          parameters['stim_pos_y']),
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
        preview_only=True,
        preview_inquiry_length=parameters['preview_inquiry_length'],
        preview_inquiry_progress_method=parameters['preview_inquiry_progress_method'],
        preview_inquiry_key_input=parameters['preview_inquiry_key_input'],
        preview_inquiry_isi=parameters['preview_inquiry_isi'])

    return CalibrationDisplay(
        window,
        static_clock,
        experiment_clock,
        stimuli,
        task_bar,
        info,
        preview_inquiry=preview_inquiry,
        trigger_type=parameters['trigger_type'],
        space_char=parameters['stim_space_char'],
        full_screen=parameters['full_screen'])
