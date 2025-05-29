"""VEP Calibration task-related code"""
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, NamedTuple

from psychopy import core, visual  # type: ignore

from bcipy.acquisition.multimodal import ClientManager
from bcipy.config import (DEFAULT_EVIDENCE_PRECISION, SESSION_DATA_FILENAME,
                          SESSION_SUMMARY_FILENAME, TRIGGER_FILENAME,
                          WAIT_SCREEN_MESSAGE)
from bcipy.display import InformationProperties, VEPStimuliProperties
from bcipy.display.components.layout import centered
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.paradigm.vep.codes import DEFAULT_FLICKER_RATES
from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.display.paradigm.vep.layout import BoxConfiguration
from bcipy.helpers.clock import Clock
# from bcipy.helpers.copy_phrase_wrapper import CopyPhraseWrapper
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.symbols import BACKSPACE_CHAR, alphabet, SPACE_CHAR
from bcipy.helpers.task import (construct_triggers, fake_vep_decision,
                                get_device_data_for_decision, get_user_input,
                                relative_triggers, target_info,
                                trial_complete_message)
from bcipy.helpers.triggers import TriggerType
from bcipy.task.base_calibration import Inquiry
from bcipy.task.paradigm.vep.stim_generation import \
    generate_vep_calibration_inquiries
    
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, convert_timing_triggers,
                                    offset_label)
from bcipy.language.main import LanguageModel
from bcipy.language.model.ambiguous import AmbiguousLanguageModel
from bcipy.language.main import ResponseType
from bcipy.signal.model.vep_signal_model import VEPSignalModel
from bcipy.task import Task
from bcipy.task.control.evidence import VEPEvaluator
from bcipy.task.data import EvidenceType, Inquiry, Session
import numpy as np

class Decision(NamedTuple):
    """Represents the result of evaluating evidence.

    Attrs
    -----
    - selection : selected target box index
    - spelled_text : spelled text resulting from the decision.
    - group_sequence : current group sequence resulting from the decision.
    """
    selection: int
    spelled_text: str
    group_sequence: List[int]


class VEPCopyPhraseTask(Task):
    """VEP Copy Phrase Task.

    A task begins setting up variables --> initializing eeg -->
        awaiting user input to start -->
        setting up stimuli --> highlighting inquiries -->
        saving data

    Parameters
    ----------
        win : object,
            display window to present visual stimuli.
        daq : object,
            data acquisition object initialized for the desired protocol
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        file_save : str,
            path location of where to save data from the session
        signal_models : list of trained signal models.
        language_model: object,
            trained language model.
        fake : boolean, optional - OVERWRITTEN
            boolean to indicate whether this is a fake session or not.
            this value is overwritten by parameters - fake_selection
    """

    TASK_NAME = 'VEP Copy Phrase Task'
    MODE = 'VEP'

    # NEEDS REVIEW
    PARAMETERS_USED = [
        'time_fixation', 'time_flash', 'time_prompt', 'trial_window',
        'font', 'fixation_color', 'trigger_type',
        'filter_high', 'filter_low', 'filter_order', 'notch_filter_frequency', 'down_sampling_rate', 'prestim_length',
        'is_txt_stim', 'show_feedback', 'feedback_duration',
        'show_preview_inquiry', 'preview_inquiry_isi',
        'spelled_letters_count',
        'stim_color', 'stim_height', 'stim_jitter', 'stim_length', 'stim_number',
        'stim_order', 'stim_pos_x', 'stim_pos_y', 'stim_space_char', 'target_color',
        'task_buffer_length', 'task_color', 'task_height', 'task_text',
        'info_pos_x', 'info_pos_y', 'info_color', 'info_height', 'info_text', 'info_color', 'info_height', 'info_text',
    ]

    def __init__(
            self,
            win: visual.Window,
            daq: ClientManager,
            parameters: Parameters,
            file_save: str,
            signal_model: VEPSignalModel,
            language_model: LanguageModel,
            fake: bool) -> None:
        super(VEPCopyPhraseTask, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.window = win
        self.daq = daq
        self.parameters = parameters
        
        # Ideally a parameter
        self.num_boxes = 8

        # Ideally not hard-coded
        self.mseq_length = 127
        self.refresh_rate = 60

        # I don't even think we use this for non-shuffle-speller vep
        self.box_colors = [
            '#00FF80', '#FFFFB3', '#CB99FF', '#FB8072', '#80B1D3', '#FF8232', '#FF8232', '#FF8232'
        ]

        # self.validate_parameters()

        self.static_clock = core.StaticPeriod(
            screenHz=self.window.getActualFrameRate())
        self.experiment_clock = Clock()
        self.start_time = self.experiment_clock.getTime()

        self.alp = alphabet(self.parameters)

        self.evidence_evaluator = self.init_evidence_evaluator(signal_model)

        self.evidence_types = [EvidenceType.LM]
        self.evidence_types.append(self.evidence_evaluator.produces)

        self.file_save = file_save

        self.trigger_handler = TriggerHandler(self.file_save, TRIGGER_FILENAME, FlushFrequency.EVERY)
        self.session_save_location = f"{self.file_save}/{SESSION_DATA_FILENAME}"
        self.copy_phrase = parameters['task_text']

        self.fake = parameters['fake_selections']
        self.language_model = AmbiguousLanguageModel(response_type=ResponseType.WORD, symbol_set=alphabet(), completions=parameters['vep_completions'])
        self.signal_model = signal_model
        self.evidence_precision = DEFAULT_EVIDENCE_PRECISION

        # self.feedback = VisualFeedback(
        #     self.window,
        #     {'feedback_font': self.parameters['font'],
        #      'feedback_color': self.parameters['info_color'],
        #      'feedback_pos_x': self.parameters['info_pos_x'],
        #      'feedback_pos_y': self.parameters['info_pos_y'],
        #      'feedback_stim_height': self.parameters['info_height'],
        #      'feedback_duration': self.parameters['feedback_duration']},
        #     self.experiment_clock)

        self.setup()

        self.vep = self.init_display()

    def init_evidence_evaluator(self,
                                signal_model: VEPSignalModel) -> VEPEvaluator:
        """Initializes the evidence evaluators from the provided signal models.

        Returns a list of evaluators for active devices. Raises an exception if
        more than one evaluator provides the same type of evidence.
        """

        evaluator = VEPEvaluator(signal_model)
        content_type = evaluator.consumes
        evidence_type = evaluator.produces
        if content_type in self.daq.active_device_content_types:
            return evaluator
        else:
            self.logger.info(
                f"SignalModel not used: there is no active device of type: {content_type}"
            )
            return None

    def setup(self) -> None:
        """Initialize/reset parameters used in the execute run loop."""

        self.spelled_text = str(
            self.copy_phrase[0:self.starting_spelled_letters()])
        self.last_selection = ''
        self.inq_counter = 0
        self.session = Session(
            save_location=self.file_save,
            task='Copy Phrase',
            mode=self.MODE,
            symbol_set=self.alp,
            decision_threshold=self.parameters['decision_threshold'])
        self.write_session_data()

        # self.init_copy_phrase_task()

    def init_display(self) -> VEPDisplay:
        """Initialize the display"""
        return _init_vep_display(self.parameters, self.window,
                                self.experiment_clock, self.alp,
                                self.box_colors)
                                
    def starting_spelled_letters(self) -> int:
        """Number of letters already spelled at the start of the task."""
        spelled_letters_count = self.parameters['spelled_letters_count']
        if spelled_letters_count > len(self.copy_phrase):
            self.logger.info('Already spelled letters exceeds phrase length.')
            spelled_letters_count = 0
        return spelled_letters_count

    # def init_copy_phrase_task(self) -> None:
    #     """Initialize the CopyPhraseWrapper.

    #     Returns:
    #     --------
    #     initialized CopyPhraseWrapper
    #     """

    #     self.copy_phrase_task = CopyPhraseWrapper(
    #         self.parameters['min_inq_len'],
    #         self.parameters['max_inq_per_series'],
    #         lmodel=self.language_model,
    #         alp=self.alp,
    #         evidence_names=self.evidence_types,
    #         task_list=[(str(self.copy_phrase), self.spelled_text)],
    #         is_txt_stim=self.parameters['is_txt_stim'],
    #         stim_timing=[
    #             self.parameters['time_fixation'], self.parameters['time_flash']
    #         ],
    #         stim_length=self.parameters['stim_length'])

    def user_wants_to_continue(self) -> bool:
        """Check if user wants to continue or terminate.

        Returns
        -------
        - `True` to continue
        - `False` to finish the task.
        """
        should_continue = get_user_input(
            self.vep,
            WAIT_SCREEN_MESSAGE,
            self.parameters['stim_color'],
            first_run=self.first_run)
        if not should_continue:
            self.logger.info('User wants to exit.')
        return should_continue

    def wait(self, seconds: Optional[float] = None) -> None:
        """Pause for a time.

        Parameters
        ----------
        - seconds : duration of time to wait; if missing, defaults to the
        value of the parameter `'task_buffer_length'`
        """
        seconds = seconds or self.parameters['task_buffer_length']
        core.wait(seconds)

    def present_inquiry(self) -> List[Tuple[str, float]]:
        """Present the given inquiry and return the trigger timing info.

        Parameters
        ----------
        None

        Returns
        -------
        stim_times

        - stim_times : list of tuples representing the stimulus and time that
        it was presented relative to the experiment clock. Non-stim triggers
        may be also be included in the list ('calibration', etc).
        """
        # Update task state and reset the static
        self.vep.update_task_bar(self.spelled_text)
        self.vep.draw_static()
        self.window.flip()

        self.wait()

        # # Setup the new stimuli
        # self.rsvp.schedule_to(stimuli=inquiry_schedule.stimuli[0],
        #                       timing=inquiry_schedule.durations[0],
        #                       colors=inquiry_schedule.colors[0]
        #                       if self.parameters['is_txt_stim'] else None)

        stim_times = self.vep.do_inquiry()

        return stim_times

    def check_stop_criteria(self) -> bool:
        """Returns True if experiment is currently within params and the task
        should continue.
        """
        if self.copy_phrase == self.spelled_text.strip(SPACE_CHAR):
            self.vep.update_task_bar(self.spelled_text)
            self.vep.draw_static()
            self.window.flip()
            self.logger.info('Spelling complete')
            return False

        if (self.inq_counter + 1) >= self.parameters['max_inq_len']:
            self.logger.info('Max tries exceeded: to allow for more tries'
                             ' adjust the Maximum inquiry Length '
                             '(max_inq_len) parameter.')
            return False

        if self.session.total_time_spent >= (self.parameters['max_minutes'] *
                                             60):
            self.logger.info('Max time exceeded. To allow for more time '
                             'adjust the max_minutes parameter.')
            return False

        if self.session.total_number_decisions >= self.parameters['max_selections']:
            self.logger.info('Max number of selections reached '
                             '(configured with the max_selections parameter)')
            return False

        return True

    def execute(self) -> str:
        """Executes the task.

        Returns
        -------
        data save location (triggers.txt, session.json)
        """
        self.logger.info('Starting Copy Phrase Task!')
        run = True
        self.wait()  # buffer for data processing

        while run and self.user_wants_to_continue():

            self.vep.update_word_predictions(self.get_top_predictions())

            stim_times = self.present_inquiry()

            self.write_trigger_data(stim_times, self.first_run)
            self.wait()

            evidence = self.compute_device_evidence(stim_times)
            decision = self.evaluate_evidence(evidence)

            # data = self.new_data_record(stim_times,
            #                             target_stimuli,
            #                             current_text=self.spelled_text,
            #                             decision=decision,
            #                             evidence_types=evidence_types)
            # self.update_session_data(data,
            #                          save=True,
            #                          decision_made=decision.decision_made)

            # if decision.decision_made:
                # self.show_feedback(decision.selection,
                #                    (decision.selection == target_stimuli))
                # self.spelled_text = decision.spelled_text
                # self.current_inquiry = self.next_inquiry()

            # else:
                # self.current_inquiry = decision.new_inq_schedule

            run = self.check_stop_criteria()
            self.inq_counter += 1

        self.exit_display()
        self.write_offset_trigger()

        self.session.task_summary = TaskSummary(
            self.session, self.parameters['show_preview_inquiry'],
            self.parameters['preview_inquiry_progress_method'],
            self.trigger_handler.file_path).as_dict()
        self.write_session_data()

        # Evidence is not recorded in the session when using fake decisions.
        if self.parameters['summarize_session'] and self.session.has_evidence():
            session_excel(session=self.session,
                          excel_file=f"{self.file_save}/{SESSION_SUMMARY_FILENAME}")

        # Wait some time before exiting so there is trailing eeg data saved
        self.wait()

        return self.file_save

    def evaluate_evidence(self, evidence: List[Tuple[EvidenceType, List[float]]]) -> Decision:
        """Uses the `copy_phrase_task` parameter to evaluate the provided
        evidence and attempt a decision.

        Modifies
        --------
        - self.copy_phrase_task
        """
        # if self.fake:
        #     _, spelled, _ = fake_copy_phrase_decision(self.copy_phrase,
                                                    #   self.next_target(),
                                                    #   self.spelled_text)
            # Reset the stoppage criteria by forcing the commit to a decision.
            # self.copy_phrase_task.decision_maker.do_series()
            # In fake mode, only the LM is providing evidence, so the decision
            # made is the highest symbol predicted. Override this state
            # self.copy_phrase_task.decision_maker.update(spelled)

            # In fake mode, all inquiries result in a selection.
            # TODO: Make this do something that makes sense
            # return Decision(selection=0,
            #                 spelled_text=spelled,
            #                 group_sequence=[0])

        # decision_made, new_sti = self.copy_phrase_task.decide()
        # spelled_text = self.copy_phrase_task.decision_maker.displayed_state
        if self.fake:
            selection_options = ["ABCDE", "FGHIJKLM", "NOPQR", "STUVWXYZ", 
                                 "Mode Switch", self.vep.word1, self.vep.word2, "Backspace"]
            selection = fake_vep_decision(selection_options, self.vep.task_bar.task_text, self.vep.task_bar.spelled_text, self.vep.chosen_boxes)
        else:
            # We only have a single type of evidence, so just use the first tuple in the list
            # Element 0 is the evidence type, element 1 is the evidence itself
            selection = np.argmax(evidence[0][1])

        self.spelled_text, group_sequence = self.vep.select(selection)
        
        return Decision(selection, self.spelled_text, group_sequence)

    # def add_evidence(self, stim_times: List[List]) -> List[EvidenceType]:
    #     """Add all evidence used to make a decision.

    #     Evaluates evidence from various sources (button press, devices,
    #     language model) and adds it to the CopyPhraseWrapper for evaluation.

    #     Parameters
    #     ----------
    #     - stim_times : list of stimuli returned from the display

    #     Returns
    #     -------
    #     list of evidence types added

    #     Modifies
    #     --------
    #     - self.copy_phrase_task
    #     """
    #     evidences = self.compute_device_evidence(stim_times)

    #     evidence_types = []
    #     for evidence in evidences:
    #         if evidence:
    #             evidence_type, probs = evidence
    #             evidence_types.append(evidence_type)
    #             self.copy_phrase_task.add_evidence(evidence_type, probs)
    #     if self.session.latest_series_is_empty():
    #         evidence_types.append(EvidenceType.LM)
    #     return evidence_types

    def compute_device_evidence(
            self,
            stim_times: List[List]) -> List[Tuple[EvidenceType, List[float]]]:
        """Get inquiry data from all devices and evaluate the evidence, but
        don't yet attempt a decision.

        Parameters
        ----------
        - stim_times : list of stimuli returned from the display

        Returns
        -------
            list of (evidence type, evidence) tuples
        """
        if self.fake:
            return []

        # currently prestim_length is used as a buffer for filter application
        post_stim_buffer = int(self.parameters.get("task_buffer_length") / 2)
        prestim_buffer: float = self.parameters['prestim_length']
        window_length = self.mseq_length / self.refresh_rate
        inquiry_timing = self.stims_for_decision(stim_times)

        # update the inquiry timing list (stim, time) based on the trial window first time value (not needed for vep?)
        # inquiry_timing = [(stim, time + trial_window[0]) for stim, time in inquiry_timing]

        # Get all data at once so we don't redundantly query devices which are
        # used in more than one signal model.
        device_data = get_device_data_for_decision(
            inquiry_timing=inquiry_timing,
            daq=self.daq,
            prestim=prestim_buffer,
            poststim=post_stim_buffer + window_length)

        triggers = relative_triggers(inquiry_timing, prestim_buffer)
        # we assume all are nontargets at this point
        # labels = ['nontarget'] * len(triggers)
        # letters, times, filtered_labels = self.copy_phrase_task.letter_info(
        #     triggers, labels)

        evidences = []
        probs = self.evidence_evaluator.evaluate(
            raw_data=device_data[self.evidence_evaluator.consumes],
            stim_time=inquiry_timing[0][1],
            window_length=window_length)
        evidences.append((self.evidence_evaluator.produces, probs))

        return evidences

    def stims_for_decision(self, stim_times: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """The stim_timings from the display may include non-stimulus timings such as PROMPT.
         method extracts only the data used to decide the target.

        Parameters
        ----------
        - stim_times : list of [stim, clock_time] pairs returned from display.

        Returns
        -------
        stim times where the label is STIMULATE.
        """
        return [
            timing for timing in stim_times if timing[0] == "STIMULATE"
        ]

    def get_top_predictions(self) -> List[str]:
        """Top two word predictions are generated from language model"""

        context = self.vep.task_bar.spelled_text
        group_sequence = [str(i) for i in self.vep.chosen_boxes]

        # Ensure valid group_sequence
        # if not group_sequence or len(group_sequence) == 0:
            # return ["", ""]  #returns empty predictions to stop crash

        if not group_sequence:
            group_sequence = []

        try:
            predictions = self.language_model.predict(list(context), list(group_sequence))

            if not predictions:
                return ["", ""]

            #Retrieve only the predicted words
            final_prediction = [word for word, _ in predictions]

            return final_prediction[:2] if len(final_prediction) >= 2 else final_prediction + [""]

        except Exception as e:
            return ["", ""]


    def init_inquiry_generator(self) -> Iterator[Inquiry]:
        """Initializes a generator that returns inquiries to be presented."""
        parameters = self.parameters
        schedule = generate_vep_calibration_inquiries(
            alp=self.alp,
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

    

    def write_session_data(self) -> None:
        """Save session data to disk."""
        if self.session:
            session_file = _save_session_related_data(
                self.session_save_location,
                self.session.as_dict())
            session_file.close()

    def write_offset_trigger(self) -> None:
        """Append the offset to the end of the triggers file.
        """
        # To help support future refactoring or use of lsl timestamps only
        # we write only the sample offset here.
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix='daq_sample_offset')
            time = client.offset(self.vep.first_stim_time)
            triggers.append(Trigger(label, TriggerType.SYSTEM, time))

        self.trigger_handler.add_triggers(triggers)
        self.trigger_handler.close()

    def write_trigger_data(self, timing: List[Tuple[str, float]],
                           first_run) -> None:
        """Write Trigger Data.

        Using the timing provided from the display and calibration information from the data acquisition
        client, write trigger data in the correct format.

        *Note on offsets*: we write the full offset value which can be used to transform all stimuli to the time since
            session start (t = 0) for all values (as opposed to most system clocks which start much higher).
            We do not write the calibration trigger used to generate this offset from the display.
            See MatrixDisplay._trigger_pulse() for more information.
        """
        if first_run:
            triggers = []
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = client.offset(self.vep.first_stim_time
                                     ) - self.vep.first_stim_time
                triggers.append(Trigger(label, TriggerType.OFFSET, time))
            self.trigger_handler.add_triggers(triggers)

        # make sure triggers are written for the inquiry
        self.trigger_handler.add_triggers(
            convert_timing_triggers(timing, timing[0][0], self.trigger_type))

    def trigger_type(self, symbol: str, target: str,
                     index: int) -> TriggerType:
        if index == 0:
            return TriggerType.PROMPT
        else:
            return TriggerType.STIMULATE

    def name(self) -> str:
        return self.TASK_NAME

    @property
    def first_run(self) -> bool:
        """First run.

        Determines whether it is the first inquiry presentation / run.
        """
        return self.inq_counter == 0

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


def _init_vep_display(parameters: Parameters, window: visual.Window,
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
    # colors = [
    #     parameters['fixation_color'], *box_colors
    # ]
    stim_props = VEPStimuliProperties(
        stim_font=parameters['font'],
        stim_pos=box_config.positions,
        stim_height=0.1,
        timing=timing,
        stim_color=box_colors,
        inquiry=[],
        stim_length=1,
        animation_seconds=parameters['time_vep_animation'])

    task_bar = CopyPhraseTaskBar(window,
                                  task_text=parameters['task_text'],
                                  spelled_text=parameters['task_text'][:parameters['spelled_letters_count']],
                                  colors=[parameters['task_color']],
                                  font=parameters['font'],
                                  height=parameters['task_height']+ 0.01)

    # issue #186641183 ; determine a better configuration strategy for flicker

    return VEPDisplay(window,
                      experiment_clock,
                      stim_props,
                      task_bar,
                      info,
                      symbol_set=symbol_set,
                      box_config=box_config,
                      flicker_rates=DEFAULT_FLICKER_RATES,
                      calibration_mode=False)


